import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import json
from transformers import BartTokenizer
import random

from modeling_bart import BartScorer
from modeling_pegasus import PegasusScorer
import torch.nn.functional as F


def RankingLoss(args, is_val, div_num, sent_rouges, sent_length, token_level_score, score, gold_score, idx=None):
    TotalLoss = 0
    SentLoss =0
    SumLoss = 0
    correct = 0

    ###########################################################################################
    # Sentence loss
    
    min_threshold=0
    if is_val: min_threshold=0
    max_threshold=10
    
    sent_scores=[]
    n = token_level_score.size(1)
    for i in range(n): #summary
        sent_score=[]
        start_position=0
        for j in range(len(sent_length[0][i])): #sentence
            end_position=start_position+sent_length[0][i][j]
            sent_score.append(token_level_score[0][i][start_position:end_position].sum(-1)/((end_position-start_position)** args.sent_length_penalty)) 
            start_position=end_position
        sent_score=torch.stack(sent_score)
        sent_scores.append(sent_score)

    summaries_sents=[]
    for i in range(n): #summary
        summary_sents=[]
        for j in range(len(sent_scores[i])): #sentence
            summary_sents.append([sent_rouges[0][i][j],sent_scores[i][j]]) 
        summary_sents = sorted(summary_sents, key=lambda x:x[0], reverse=True)
        summaries_sents.append(summary_sents)
    
    #candidate 별로 random 하게 inter-sentence sampling
    if args.inter: 
        for i in range(n): #summary   
            sum_idx_list=torch.arange(n).tolist()
            sum_idx_list.remove(i) 
            selected_sum_idx_list=random.sample(sum_idx_list,args.random_sent_num)
            for selected_sum_idx in selected_sum_idx_list:
                sent_idx_list=torch.arange(len(summaries_sents[selected_sum_idx])).tolist()
                selected_sent_idx= random.choice(sent_idx_list)
                summaries_sents[i].append(summaries_sents[selected_sum_idx][selected_sent_idx])
                summaries_sents[i]=sorted(summaries_sents[i], key=lambda x:x[0], reverse=True)

    sorted_sent_scores=[]
    for i in range(n): #summary
        sorted_sent_score=[]
        for j in range(len(summaries_sents[i])): #sentence
            sorted_sent_score.append(summaries_sents[i][j][1])
        sorted_sent_scores.append(sorted_sent_score)

    sorted_sent_rouges=[]
    for i in range(n): #summary
        sorted_sent_rouge=[]
        for j in range(len(summaries_sents[i])): #sentence
            sorted_sent_rouge.append(summaries_sents[i][j][0])
        sorted_sent_rouges.append(sorted_sent_rouge)
        
    ###########################################################################################
    # Sentence-level margin ranking loss
    for i in range(1,n): #summary
        sent_loss=0
        sent_pair_count=0
        
        for j in range(1,len(summaries_sents[i])): #sentence
            pos_sent_score=torch.stack(sorted_sent_scores[i][:-j])
            neg_sent_score=torch.stack(sorted_sent_scores[i][j:])
            
            pos_sent_rouge=torch.Tensor(sorted_sent_rouges[i][:-j]).to(args.gpuid)
            neg_sent_rouge=torch.Tensor(sorted_sent_rouges[i][j:]).to(args.gpuid)
    
            rouge_gap=pos_sent_rouge-neg_sent_rouge
            
            check=torch.where(rouge_gap>min_threshold,1,0)
            check=torch.mul(check,torch.where(rouge_gap<max_threshold,1,0))
            
            ones = torch.ones_like(pos_sent_score)
            
            for k in range(pos_sent_score.size(0)):
                sent_loss_func = torch.nn.MarginRankingLoss(args.sent_margin * (rouge_gap[k]*10)) #0.02 
                loss = sent_loss_func(pos_sent_score[k], neg_sent_score[k], ones[k]) 
                
                loss=torch.mul(loss, check[k]) #####
                
                sent_loss+=loss
                if check[k]:
                    sent_pair_count+=1

        TotalLoss += sent_loss/max(sent_pair_count,1) *args.sent_loss_weight
        SentLoss += sent_loss/max(sent_pair_count,1)*args.sent_loss_weight

    for i in range(len(sorted_sent_scores)): # summary
        summary_sentence_scores=torch.stack(sorted_sent_scores[i])
        if torch.min(summary_sentence_scores)==summary_sentence_scores[-1]:
            correct+=1
    acc=correct/len(sorted_sent_scores)*100
    soft_acc=0
    
    ###########################################################################################
    
    # Candidate summary loss
    n = score.size(1)
    for i in range(1, n):
        pos_score = score[:, :-i]
        neg_score = score[:, i:]
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        ones = torch.ones_like(pos_score)
        # loss_func = torch.nn.MarginRankingLoss(args.margin * i, reduction='none')
        loss_func = torch.nn.MarginRankingLoss(args.margin * i)
        
        loss = loss_func(pos_score, neg_score, ones)
        
        TotalLoss += loss
        SumLoss += loss
        
    #TotalLoss=TotalLoss/n
    #TotalLoss=TotalLoss/n
    
    print("SumLoss: ",SumLoss)
    print("SentLoss: ",SentLoss)
    print("TotalLoss: ",TotalLoss)
    
    return TotalLoss, SumLoss, SentLoss, acc, soft_acc
    
    ###########################################################################################
    
    # Gold summary loss
    # pos_score = gold_score.unsqueeze(-1).expand_as(score)
    # neg_score = score
    # pos_score = pos_score.contiguous().view(-1)
    # neg_score = neg_score.contiguous().view(-1)
    # ones = torch.ones_like(pos_score)
    # loss_func = torch.nn.MarginRankingLoss(args.gold_margin)
    #TotalLoss += args.gold_weight * loss_func(pos_score, neg_score, ones)
    #SumLoss += args.gold_weight * loss_func(pos_score, neg_score, ones)
        
    return TotalLoss, SumLoss, SentLoss, acc, soft_acc


def ValRankingLoss(args, output_path, div_num, idx, sent_length, token_level_score, score):  #####
    with open('../data/cnndm/diverse/new_val/'+str(idx)+'.json', 'r') as f:
        json_data = json.load(f) 
        
    sent_scores=[]
    n = token_level_score.size(1)
    for i in range(1,n): #summary
        sent_score=[]
        start_position=0
        for j in range(len(sent_length[0][i])): #sentence
            end_position=start_position+sent_length[0][i][j]
            sent_score.append(token_level_score[0][i][start_position:end_position].sum(-1)/((end_position-start_position)** args.sent_length_penalty)) # average with length penalty
            start_position=end_position
        sent_score_eval=torch.stack(sent_score).tolist()
        sent_score=torch.stack(sent_score)
        
        json_data["candidates"][i-1].append(sent_score_eval) 
        json_data["candidates_untok"][i-1].append(sent_score_eval) 
        json_data["candidates"][i-1].append(score[0][i-1].item()) 
        json_data["candidates_untok"][i-1].append(score[0][i-1].item()) 
        sent_scores.append(sent_score)

    with open(output_path+"/"+str(idx)+'.json', 'w', encoding='utf-8') as make_file:
        json.dump(json_data, make_file, indent="\t") 

        
def EvalRankingLoss(args, output_path, div_num, idx, sent_length, token_level_score, score):  #####
    with open('../data/cnndm/diverse/new_test/'+str(idx)+'.json', 'r') as f:
        json_data = json.load(f) 
        
    sent_scores=[]
    n = token_level_score.size(1)
    for i in range(1,n): #summary
        sent_score=[]
        start_position=0
        for j in range(len(sent_length[0][i])): #sentence
            end_position=start_position+sent_length[0][i][j]
            sent_score.append(token_level_score[0][i][start_position:end_position].sum(-1)/((end_position-start_position)** args.sent_length_penalty)) # average with length penalty
            start_position=end_position
        sent_score_eval=torch.stack(sent_score).tolist()
        sent_score=torch.stack(sent_score)
        
        json_data["candidates"][i-1].append(sent_score_eval) 
        json_data["candidates_untok"][i-1].append(sent_score_eval) 
        json_data["candidates"][i-1].append(score[0][i-1].item()) 
        json_data["candidates_untok"][i-1].append(score[0][i-1].item()) 
        sent_scores.append(sent_score)

    with open(output_path+"/"+str(idx)+'.json', 'w', encoding='utf-8') as make_file:
        json.dump(json_data, make_file, indent="\t") 


class BRIO(nn.Module):
    def __init__(self, args, pad_token_id):
        super(BRIO, self).__init__()
        if args.is_pegasus:
            self.model = PegasusScorer.from_pretrained(args.model_type, cache_dir="./local_cache")
        else:
            self.model = BartScorer.from_pretrained(args.model_type, cache_dir="./local_cache")
        self.pad_token_id = pad_token_id
        
        self.tokenizer= BartTokenizer.from_pretrained(args.model_type)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, args, article_id, candidate_id, decoder_sent_mask, require_gold=True): 
        batch_size = article_id.size(0)
        
        input_mask = article_id != self.pad_token_id #####
        cand_mask = candidate_id != self.pad_token_id #####
        cand_mask[:, :, 0] = 1 #####
        
        output = self.model(
            input_ids=article_id, 
            attention_mask=input_mask,
            decoder_input_ids=candidate_id, 
            decoder_attention_mask=cand_mask,
            output_hidden_states=True,
            decoder_sent_mask=decoder_sent_mask, #####
            layer_apply_sent_mask=args.layer_apply_sent_mask #####
            )
        
        output = output[0]  # [b_s * cand_num, seq_len, word_dim]
        output = output.view(batch_size, -1, output.size(1), output.size(2)) # [b_s, cand_num, seq_len, word_dim]
        
        probs = output[:, 0] # gold summary (for mle loss)
        
        output = output[:, :, :-1]  # truncate last token #####
        candidate_id = candidate_id[:, :, 1:]  # truncate start token #####
        cand_mask = candidate_id != self.pad_token_id
        candidate_id = candidate_id.unsqueeze(-1)
        
        if args.softmax:
            _output = F.softmax(output, dim=3)

        else:
            _output = F.log_softmax(output, dim=3)
            
        scores = torch.gather(_output, 3, candidate_id).squeeze(-1)  # [bz, cand_num, seq_len]
                    
        cand_mask = cand_mask.float()
        
        ###########################################################################################
        token_level_scores= scores # [b_s, gold_num+cand_num, seq_len] # token 별 생성 확률 값
        
        scores = torch.mul(scores, cand_mask).sum(-1) / ((cand_mask.sum(-1)) ** args.length_penalty) # [b_s, gold_num+cand_num] # summary 별 생성 확률 값
        
        div_num=(cand_mask.sum(-1)) ** args.length_penalty
        
        if require_gold:
            output = {'score': scores[:, 1:], "summary_score": scores[:, 0], "probs": probs, "token_level_scores":token_level_scores, "div_num":div_num} 
        else:
            output = {'score': scores, "probs": probs, "token_level_scores":token_level_scores, "div_num":div_num} 
        ###########################################################################################
        
        return output

    def scoring_mode(self):
        self.model.model.scoring_mode()