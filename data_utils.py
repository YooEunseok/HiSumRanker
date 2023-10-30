from torch.utils.data import Dataset
import torch
import os
import json
torch.set_printoptions(profile="full")


class BrioDataset(Dataset):
    def __init__(self, datapath, tokenizer, max_len, total_len, is_test, is_sorted, is_pegasus, is_untok=True):
        self.datapath=datapath
        self.num = len(os.listdir(datapath))
        self.tokenizer=tokenizer
        self.max_len = max_len
        self.total_len = total_len
        self.is_test = is_test
        self.is_sorted = is_sorted
        self.is_pegasus = is_pegasus
        self.is_untok = is_untok
        self.cand_num = 16

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        with open(os.path.join(self.datapath, "%d.json"%idx), "r") as f:
            data = json.load(f)
            
        # print(idx)

        # Article
        article = data["article_untok"]
        src_txt = " ".join(article)
        
        # Article tokenize
        src = self.tokenizer.batch_encode_plus([src_txt], max_length=self.total_len, return_tensors="pt", pad_to_max_length=False, truncation=True) #####
        src_input_ids = src["input_ids"].squeeze(0)
        
        # Abstract
        abstract = data["abstract_untok"]
        
        # Candidate
        candidates = data["candidates_untok"][:self.cand_num]
        _candidates = data["candidates"][:self.cand_num] #
        data["candidates"] = _candidates #
        if self.is_sorted:
            candidates = sorted(candidates, key=lambda x:x[1], reverse=True)
            _candidates = sorted(_candidates, key=lambda x:x[1], reverse=True) #
            data["candidates"] = _candidates #
        
        # Abstract, Candidate tokenize
        cand_txt = [" ".join(abstract)] + [" ".join(x[0]) for x in candidates] #####
        cand = self.tokenizer.batch_encode_plus(cand_txt, max_length=self.max_len, return_tensors="pt", pad_to_max_length=False, truncation=True, padding=True) #####
        candidate_ids = cand["input_ids"]
        
        
        # Summary-level Score
        score=[x[1] for x in candidates]
        
        ###########################################################################################
        # Sentence tokenize
        
        sent_cand_txt=[]
        sent_lens=[]

        gold_txt=[" "+sent for sent in abstract] # sentences in one candidate summary # sentence의 앞에 space 붙여서 저장
        gold_txt[0]=gold_txt[0][1:] # 맨 앞 sentence에는 space 빼기
        
        sent_gold = self.tokenizer.batch_encode_plus(gold_txt, max_length=self.max_len, return_tensors="pt", pad_to_max_length=False, truncation=True, padding=True) 
        sent_gold_ids = sent_gold["input_ids"] # [sent num, seq length]

        eos_idx=(sent_gold_ids == 2).nonzero(as_tuple=True)[1] # 각 sentence 별 eos_token의 position을 저장 
        ones=torch.ones(eos_idx.shape) 
        gold_sent_len=eos_idx-ones.type(torch.int) # sos_token 제외

        sent_lens.append(gold_sent_len.tolist())

        for x in candidates:
            sent_len=[]
            
            sent_cand_txt=[" "+sent for sent in x[0]] # sentences in one candidate summary # sentence의 앞에 space 붙여서 저장
            sent_cand_txt[0]=sent_cand_txt[0][1:] # 맨 앞 sentence에는 space 빼기

            sent_cand = self.tokenizer.batch_encode_plus(sent_cand_txt, max_length=self.max_len, return_tensors="pt", pad_to_max_length=False, truncation=True, padding=True) 
            sent_candidate_ids = sent_cand["input_ids"] # [sent num, seq length]

            eos_idx=(sent_candidate_ids == 2).nonzero(as_tuple=True)[1] # 각 sentence 별 eos_token의 position을 저장 
            ones=torch.ones(eos_idx.shape) 
            sent_len=eos_idx-ones.type(torch.int) # sos_token 제외

            sent_lens.append(sent_len.tolist())
        # sent_lens: [[22, 22, 16], [19, 15, 28], [13, 14, 15, 12], [22, 22, 21], [14, 14, 28], [31, 14, 18], [13, 13, 31], [16, 14, 17, 14], [15, 11, 12, 16], [19, 15, 32], [14, 14, 31], [13, 14, 15, 12, 13], [12, 14, 31], [15, 14, 31], [16, 13, 12, 15], [17, 15, 31]]

        # Sentence-level Score
        sent_rouges=[]
        gold_rouge=[]
        for i in range(len(gold_sent_len)):
            gold_rouge.append(1)
        sent_rouges.append(gold_rouge)
        for x in candidates:
            sent_rouge=[score for score in x[2]]
            sent_rouges.append(sent_rouge)
        # sent_rouges: [[0.5294117647058824, 0.631578947368421, 0.36363636363636365], [0.42857142857142855, 0.8461538461538461, 0.23809523809523808], [0.7272727272727273, 0.8333333333333334, 0.08333333333333333, 0.18181818181818182], [0.5294117647058824, 0.631578947368421, 0.3333333333333333], [0.5454545454545454, 0.8333333333333334, 0.19047619047619047], [0.45454545454545453, 0.8333333333333334, 0.3333333333333333], [0.7272727272727273, 0.8333333333333334, 0.13636363636363635], [0.16666666666666666, 0.5454545454545454, 0.38461538461538464, 0.8333333333333334], [0.08333333333333333, 0.7, 0.8181818181818182, 0.18181818181818182], [0.42857142857142855, 0.8461538461538461, 0.25], [0.5454545454545454, 0.8333333333333334, 0.16666666666666666], [0.7272727272727273, 0.8333333333333334, 0.08333333333333333, 0.18181818181818182, 0.25], [0.5, 0.8333333333333334, 0.17391304347826086], [0.5, 0.8333333333333334, 0.17391304347826086], [0.5833333333333334, 0.16666666666666666, 0.18181818181818182, 0.4166666666666667], [0.38461538461538464, 0.5, 0.21739130434782608]]
        
        ###########################################################################################
        
        if self.is_pegasus: #####
            # add start token
            _candidate_ids = candidate_ids.new_zeros(candidate_ids.size(0), candidate_ids.size(1) + 1)
            _candidate_ids[:, 1:] = candidate_ids.clone()
            _candidate_ids[:, 0] = self.tokenizer.pad_token_id
            candidate_ids = _candidate_ids
            
        result = {
            "src_input_ids": src_input_ids, 
            "candidate_ids": candidate_ids,
            "scores": score,
            "sent_lens": sent_lens, 
            "sent_rouges": sent_rouges,
            "idx":idx
            }
        
        if self.is_test: 
            result["data"] = data
            result["idx"] = idx 
        return result


def collate_mp_brio(batch, pad_token_id, args,is_test=False):
    def pad(X, max_len=-1):
        if max_len < 0:
            max_len = max(x.size(0) for x in X)
        result = torch.ones(len(X), max_len, dtype=X[0].dtype) * pad_token_id
        for (i, x) in enumerate(X):
            result[i, :x.size(0)] = x
        return result

    src_input_ids = pad([x["src_input_ids"] for x in batch])
    
    candidate_ids = [x["candidate_ids"] for x in batch]
    max_len = max([max([len(c) for c in x]) for x in candidate_ids])
    candidate_ids = [pad(x, max_len) for x in candidate_ids]
    candidate_ids = torch.stack(candidate_ids)
    
    score= [x["scores"] for x in batch]
    score=torch.Tensor(score)
    
    idx= [x["idx"] for x in batch]
    

    ###########################################################################################
    # Decoder sent mask
    
    sent_lens= [x["sent_lens"] for x in batch] 
    sent_rouges= [x["sent_rouges"] for x in batch] 

    batch_size=candidate_ids.size(0)
    sum_num=candidate_ids.size(1)
    seq_len=candidate_ids.size(2)

    # temp_mask=torch.zeros(1,2,10,10)
    # for i in range(2,4):
    #     for j in range(2,4):
    #         temp_mask[0][1][i][j]=1
    # print(temp_mask)
    # assert 0
    
    decoder_sent_mask=None
    if args.decoder_sent_mask:

        decoder_sent_mask=torch.zeros(batch_size,sum_num,seq_len,seq_len)
        # torch.Size([1, 17, 69, 69])

        for b in range(batch_size):
            for i in range(sum_num): # summary
                start_position=1 
                for j in range(len(sent_lens[b][i])): # sentence
                    end_position=start_position+sent_lens[b][i][j]
                    for s in range(start_position,end_position):
                        decoder_sent_mask[b][i][min(s,decoder_sent_mask.size(-1)-1)][0]=1 # start position
                        decoder_sent_mask[b][i][0][min(s,decoder_sent_mask.size(-1)-1)]=1 # start position
                        for e in range(start_position,end_position):
                            decoder_sent_mask[b][i][min(s,decoder_sent_mask.size(-1)-1)][min(e,decoder_sent_mask.size(-1)-1)]=1
                    start_position=end_position

    ###########################################################################################
    
    if is_test:
        data = [x["data"] for x in batch]        
        idx = [x["idx"] for x in batch] 
        
    result = {
        "src_input_ids": src_input_ids,
        "candidate_ids": candidate_ids,
        "scores": score,
        "sent_lens": sent_lens, 
        "sent_rouges": sent_rouges,
        "decoder_sent_mask": decoder_sent_mask,
        "idx":idx
        }
    
    if is_test:
        result["data"] = data
        result["idx"] = idx
        
    return result