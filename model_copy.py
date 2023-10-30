import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import json
from transformers import BartTokenizer
import random

from modeling_bart import BartScorer
from modeling_pegasus import PegasusScorer


def RankingLoss(args, is_val, div_num, sent_rouges, sent_length, token_level_score, score, gold_score):
    TotalLoss = 0
    # SentLoss = []
    # SumLoss = []
    SentLoss =0
    SumLoss = 0
    correct = 0

    ###########################################################################################
    # Sentence loss
    
    min_threshold=0
    if is_val: min_threshold=0
    max_threshold=10
    
    #token_level_score=token_level_score[:,1:] # [b_s, cand_num+gold_num -> cand_num, seq_len] # gold 제외 
    
    # print(sent_rouges) # cand num * sent num(summary 마다 다름)    
    # [[[0.5294117647058824, 0.631578947368421, 0.36363636363636365], [0.42857142857142855, 0.8461538461538461, 0.23809523809523808], [0.7272727272727273, 0.8333333333333334, 0.08333333333333333, 0.18181818181818182], [0.5294117647058824, 0.631578947368421, 0.3333333333333333], [0.5454545454545454, 0.8333333333333334, 0.19047619047619047], [0.45454545454545453, 0.8333333333333334, 0.3333333333333333], [0.7272727272727273, 0.8333333333333334, 0.13636363636363635], [0.16666666666666666, 0.5454545454545454, 0.38461538461538464, 0.8333333333333334], [0.08333333333333333, 0.7, 0.8181818181818182, 0.18181818181818182], [0.42857142857142855, 0.8461538461538461, 0.25], [0.5454545454545454, 0.8333333333333334, 0.16666666666666666], [0.7272727272727273, 0.8333333333333334, 0.08333333333333333, 0.18181818181818182, 0.25], [0.5, 0.8333333333333334, 0.17391304347826086], [0.5, 0.8333333333333334, 0.17391304347826086], [0.5833333333333334, 0.16666666666666666, 0.18181818181818182, 0.4166666666666667], [0.38461538461538464, 0.5, 0.21739130434782608]]]
    # print(sent_length) # cand num * sent num(summary 마다 다름)    
    # [[[22, 22, 16], [19, 15, 28], [13, 14, 15, 12], [22, 22, 21], [14, 14, 28], [31, 14, 18], [13, 13, 31], [16, 14, 17, 14], [15, 11, 12, 16], [19, 15, 32], [14, 14, 31], [13, 14, 15, 12, 13], [12, 14, 31], [15, 14, 31], [16, 13, 12, 15], [17, 15, 31]]]

    # token level score를 sentence 단위로 자르고 평균 내서 sent_scores 저장
    sent_scores=[]
    n = token_level_score.size(1)
    for i in range(n): #summary
        
        sent_score=[]
        start_position=0
        for j in range(len(sent_length[0][i])): #sentence
            end_position=start_position+sent_length[0][i][j]
            sent_score.append(token_level_score[0][i][start_position:end_position].sum(-1)/((end_position-start_position)** args.sent_length_penalty)) # average with length penalty
            start_position=end_position
        sent_score=torch.stack(sent_score)
        sent_scores.append(sent_score)
    # [tensor([-1.5128, -0.3372, -0.6248], device='cuda:1', grad_fn=<StackBackward0>), tensor([-0.8002, -0.3659, -0.6998], device='cuda:1', grad_fn=<StackBackward0>), ...  

    # sent_rouges를 기준으로 (sent_rouges-sent_scores) pair들을 sorting
    summaries_sents=[]
    for i in range(n): #summary
        summary_sents=[]
        for j in range(len(sent_scores[i])): #sentence
            summary_sents.append([sent_rouges[0][i][j],sent_scores[i][j]]) # i번째 summary의 j번째 sentence
        summary_sents = sorted(summary_sents, key=lambda x:x[0], reverse=True)
        summaries_sents.append(summary_sents)
    # print(summaries_sents)
    # # [[[0.631578947368421, tensor(-0.3372, device='cuda:1', grad_fn=<SelectBackward0>)], [0.5294117647058824, tensor(-1.5128, device='cuda:1', grad_fn=<SelectBackward0>)],
    
    for i in range(n): #summary    
        sum_idx_list=torch.arange(n).tolist()
        sum_idx_list.remove(i) 
        selected_sum_idx_list=random.sample(sum_idx_list,args.random_sent_num)
        for selected_sum_idx in selected_sum_idx_list:
            sent_idx_list=torch.arange(len(summaries_sents[selected_sum_idx])).tolist()
            selected_sent_idx= random.choice(sent_idx_list)
            # print(selected_sum_idx, selected_sent_idx)
            summaries_sents[i].append(summaries_sents[selected_sum_idx][selected_sent_idx])
            summaries_sents[i]=sorted(summaries_sents[i], key=lambda x:x[0], reverse=True)
    # # print(summaries_sents[0])
    # # assert 0

    # sorted_sent_scores 저장
    sorted_sent_scores=[]
    for i in range(n): #summary
        sorted_sent_score=[]
        for j in range(len(summaries_sents[i])): #sentence
            sorted_sent_score.append(summaries_sents[i][j][1])
        sorted_sent_scores.append(sorted_sent_score)
    # print(sorted_sent_scores)
    # [[tensor(-0.3372, device='cuda:1', grad_fn=<SelectBackward0>), tensor(-1.5128, device='cuda:1', grad_fn=<SelectBackward0>), ...

    # sorted_sent_rouges 저장
    sorted_sent_rouges=[]
    for i in range(n): #summary
        sorted_sent_rouge=[]
        for j in range(len(summaries_sents[i])): #sentence
            sorted_sent_rouge.append(summaries_sents[i][j][0])
        sorted_sent_rouges.append(sorted_sent_rouge)
    # print(sorted_sent_rouges)
    # [[0.631578947368421, 0.5294117647058824, 0.36363636363636365], [0.8461538461538461, 0.42857142857142855, 0.23809523809523808], ...
    
    total_check=[]
    # Sentence-level margin ranking loss
    for i in range(n): #summary
        for j in range(1,len(summaries_sents[i])): #sentence
            # print(sorted_sent_scores[i])
            pos_sent_score=torch.stack(sorted_sent_scores[i][:-j])
            neg_sent_score=torch.stack(sorted_sent_scores[i][j:])
   
            
            pos_sent_rouge=torch.Tensor(sorted_sent_rouges[i][:-j]).to(args.gpuid)
            neg_sent_rouge=torch.Tensor(sorted_sent_rouges[i][j:]).to(args.gpuid)
    
            rouge_gap=pos_sent_rouge-neg_sent_rouge
            
            check=torch.where(rouge_gap>min_threshold,1,0)
            check=torch.mul(check,torch.where(rouge_gap<max_threshold,1,0))

            ones = torch.ones_like(pos_sent_score)
            sent_loss_func = torch.nn.MarginRankingLoss(args.sent_margin * j, reduction='none') 
            loss = sent_loss_func(pos_sent_score, neg_sent_score, ones)
            loss= torch.mul(loss, check)

            one=torch.Tensor([1])
            loss=torch.div(torch.sum(loss),torch.max(torch.sum(check),one[0])) #sentence 개수의 평균
            # print(loss)
            
            TotalLoss += loss*args.sent_loss_weight
            SentLoss += loss*args.sent_loss_weight
            
    #         total_check.append(check)
    #         SentLoss.append(loss)

    # SentLoss=torch.cat(SentLoss)
    # total_check=torch.cat(total_check)
    # SentLoss=torch.div(torch.sum(SentLoss),max(torch.sum(total_check),1)) 

    for i in range(len(sorted_sent_scores)): # summary
        summary_sentence_scores=torch.stack(sorted_sent_scores[i])
        if torch.min(summary_sentence_scores)==summary_sentence_scores[-1]:
            correct+=1
    acc=correct/len(sorted_sent_scores)*100
    soft_acc=0
    
    # ***** batch 고려 안한 것 나중에 처리해주기 *****
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
        
    #     SumLoss.append(loss)
    # SumLoss=torch.cat(SumLoss)
    # SumLoss=torch.mean(SumLoss)

    # TotalLoss = SumLoss+SentLoss*args.sent_loss_weight
    # SentLoss=SentLoss*args.sent_loss_weight
    
    print("SumLoss: ",SumLoss)
    print("SentLoss: ",SentLoss)
    print("TotalLoss: ",TotalLoss)
    
    #return TotalLoss, SumLoss, SentLoss, acc, soft_acc
    
    # Gold summary loss
    pos_score = gold_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones_like(pos_score)
    loss_func = torch.nn.MarginRankingLoss(args.gold_margin)
    #TotalLoss += args.gold_weight * loss_func(pos_score, neg_score, ones)
    #SumLoss += args.gold_weight * loss_func(pos_score, neg_score, ones)
        
    return TotalLoss, SumLoss, SentLoss, acc, soft_acc


def ValRankingLoss(args, output_path, div_num, idx, sent_length, token_level_score, score):  #####
    with open('/home/nlplab/hdd1/yoo/BRIO/data/cnndm/diverse/new_val/'+str(idx)+'.json', 'r') as f:
        json_data = json.load(f) 
        
    sent_scores=[]
    n = token_level_score.size(1)
    for i in range(1,n): #summary
        sent_score=[]
        start_position=0
        for j in range(len(sent_length[0][i])): #sentence
            end_position=start_position+sent_length[0][i][j]
            sent_score.append(token_level_score[0][i][start_position:end_position].sum(-1)/((end_position-start_position)** args.sent_length_penalty)) # average with length penalty
            # if args.length_penalty==1.0:
            #     sent_score.append(token_level_score[0][i][start_position:end_position].sum(-1)/((end_position-start_position)** args.length_penalty)) # average with length penalty
            # else:
            #     sent_score.append(token_level_score[0][i][start_position:end_position].sum(-1)/div_num[0][i])
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
    with open('/home/nlplab/hdd1/yoo/BRIO/data/cnndm/diverse/new_test/'+str(idx)+'.json', 'r') as f:
        json_data = json.load(f) 
        
    sent_scores=[]
    n = token_level_score.size(1)
    for i in range(1,n): #summary
        sent_score=[]
        start_position=0
        for j in range(len(sent_length[0][i])): #sentence
            end_position=start_position+sent_length[0][i][j]
            sent_score.append(token_level_score[0][i][start_position:end_position].sum(-1)/((end_position-start_position)** args.sent_length_penalty)) # average with length penalty
            # sent_score.append(token_level_score[0][i][start_position:end_position].sum(-1)/((end_position-start_position)* div_num[0][i])*1000) # average with length penalty
            
            # if args.length_penalty==1.0:
            #     sent_score.append(token_level_score[0][i][start_position:end_position].sum(-1)/((end_position-start_position)** args.length_penalty)) # average with length penalty
            # else:
            #     sent_score.append(token_level_score[0][i][start_position:end_position].sum(-1)/div_num[0][i])
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
        
        _output = F.log_softmax(output, dim=3)
        scores = torch.gather(_output, 3, candidate_id).squeeze(-1)  # [bz, cand_num, seq_len]
                    
        cand_mask = cand_mask.float()
        
        ###########################################################################################
        token_level_scores= scores # [b_s, gold_num+cand_num, seq_len] # token 별 생성 확률 값
        # print(token_level_scores[0][1])
        # assert 0
        
        scores = torch.mul(scores, cand_mask).sum(-1) / ((cand_mask.sum(-1)) ** args.length_penalty) # [b_s, gold_num+cand_num] # summary 별 생성 확률 값

        # scores = self.softmax(scores)
        # scores = torch.mul(scores, cand_mask).sum(-1) / ((cand_mask.sum(-1)) * 100) # [b_s, gold_num+cand_num] # summary 별 생성 확률 값
        
        div_num=(cand_mask.sum(-1)) ** args.length_penalty
        
        if require_gold:
            output = {'score': scores[:, 1:], "summary_score": scores[:, 0], "probs": probs, "token_level_scores":token_level_scores, "div_num":div_num} 
        else:
            output = {'score': scores, "probs": probs, "token_level_scores":token_level_scores, "div_num":div_num} 
        ###########################################################################################
        
        return output

    def scoring_mode(self):
        self.model.model.scoring_mode()

    # def generation_mode(self):
    #     self.model.model.generation_mode()

    # def generate(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     max_length: Optional[int] = None,
    #     min_length: Optional[int] = None,
    #     do_sample: Optional[bool] = None,
    #     early_stopping: Optional[bool] = None,
    #     num_beams: Optional[int] = None,
    #     temperature: Optional[float] = None,
    #     top_k: Optional[int] = None,
    #     top_p: Optional[float] = None,
    #     repetition_penalty: Optional[float] = None,
    #     bad_words_ids: Optional[Iterable[int]] = None,
    #     bos_token_id: Optional[int] = None,
    #     pad_token_id: Optional[int] = None,
    #     eos_token_id: Optional[int] = None,
    #     length_penalty: Optional[float] = None,
    #     no_repeat_ngram_size: Optional[int] = None,
    #     encoder_no_repeat_ngram_size: Optional[int] = None,
    #     num_return_sequences: Optional[int] = None,
    #     max_time: Optional[float] = None,
    #     decoder_start_token_id: Optional[int] = None,
    #     use_cache: Optional[bool] = None,
    #     num_beam_groups: Optional[int] = None,
    #     diversity_penalty: Optional[float] = None,
    #     prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     output_scores: Optional[bool] = None,
    #     return_dict_in_generate: Optional[bool] = None,
    #     forced_bos_token_id: Optional[int] = None,
    #     forced_eos_token_id: Optional[int] = None,
    #     remove_invalid_values: Optional[bool] = None,
    #     synced_gpus: Optional[bool] = None,
    #     **model_kwargs,
    # ):
    #     return self.model.generate(input_ids=input_ids,
    #         max_length=max_length,
    #         min_length=min_length,
    #         do_sample=do_sample,
    #         early_stopping=early_stopping,
    #         num_beams=num_beams,
    #         temperature=temperature,
    #         top_k=top_k,
    #         top_p=top_p,
    #         repetition_penalty=repetition_penalty,
    #         bad_words_ids=bad_words_ids,
    #         bos_token_id=bos_token_id,
    #         pad_token_id=pad_token_id,
    #         eos_token_id=eos_token_id,
    #         length_penalty=length_penalty,
    #         no_repeat_ngram_size=no_repeat_ngram_size,
    #         encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
    #         num_return_sequences=num_return_sequences,
    #         max_time=max_time,
    #         decoder_start_token_id=decoder_start_token_id,
    #         use_cache=use_cache,
    #         num_beam_groups=num_beam_groups,
    #         diversity_penalty=diversity_penalty,
    #         prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         output_scores=output_scores,
    #         return_dict_in_generate=return_dict_in_generate,
    #         forced_bos_token_id=forced_bos_token_id,
    #         forced_eos_token_id=forced_eos_token_id,
    #         remove_invalid_values=remove_invalid_values,
    #         synced_gpus=synced_gpus,
    #         **model_kwargs)