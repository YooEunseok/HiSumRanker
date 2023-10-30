import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
import random
from datetime import datetime
from functools import partial
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
from transformers import BartTokenizer, PegasusTokenizer
from compare_mt.rouge.rouge_scorer import RougeScorer

from data_utils import collate_mp_brio, BrioDataset
from model import RankingLoss, ValRankingLoss, EvalRankingLoss, BRIO
from label_smoothing_loss import label_smoothing_loss
from config import cnndm_setting
from check import sent_length_score_check, sent_position_score_check


def evaluation(args):
    if args.dataset == "cnndm":
        cnndm_setting(args)
    # if args.dataset == "xsum":
    #     xsum_setting(args)

    if args.is_pegasus: 
        tok = PegasusTokenizer.from_pretrained(args.model_type) 
    else: 
        tok = BartTokenizer.from_pretrained(args.model_type) 
        
    # Dataset
    test_set = BrioDataset(args.datapath+"/new_test", tok, max_len=512, total_len=args.total_len, is_test=True, is_sorted=False, is_pegasus=args.is_pegasus, is_untok=True)
    
    collate_fn = partial(collate_mp_brio, pad_token_id=tok.pad_token_id, args=args, is_test=True)
    
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    # Model
    model = BRIO(args, tok.pad_token_id) 
    model = model.to(args.gpuid)
    model.load_state_dict(torch.load(os.path.join("./checkpoints", args.model_pt), map_location=f'cuda:{args.gpuid}')) 
    model.eval()
    
    # Output 
    model_name=args.model_pt.split("/")[-2]
    output_path = "/home/nlplab/hdd1/yoo/BRIO/data/cnndm/diverse/test_output/"+model_name
    if os.path.exists(output_path):
        assert "save path exist"
    os.makedirs(output_path,exist_ok=True)

    # Metric
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    rouge1, rouge2, rougeLsum = 0, 0, 0
    cnt = 0
    
    eval_count=0
    
    model.scoring_mode() #####
    
    is_val=1
    # Evaluation
    with torch.no_grad():
        for batch in tqdm(test_dataloader):   
            for b in batch:
                if b != "data" and b!="sent_lens" and b!="sent_rouges" and b!="idx" and batch[b] is not None:
                    batch[b] = batch[b].to(args.gpuid)
            samples = batch["data"]
            
            output = model(args, batch["src_input_ids"], batch["candidate_ids"], batch["decoder_sent_mask"])
            similarity, token_level_scores = output['score'], output['token_level_scores']
            similarity = similarity.cpu().numpy()
            max_ids = similarity.argmax(1)
            div_num=output["div_num"]

            

            EvalRankingLoss(args, output_path, div_num, eval_count, batch["sent_lens"], token_level_scores, similarity)            
            eval_count+=1

            
            for j in range(similarity.shape[0]):
                sample = samples[j]
                sents = sample["candidates"][max_ids[j]][0]
                score = rouge_scorer.score("\n".join(sample["abstract"]), "\n".join(sents))
                rouge1 += score["rouge1"].fmeasure
                rouge2 += score["rouge2"].fmeasure
                rougeLsum += score["rougeLsum"].fmeasure
                cnt += 1
                
        rouge1 = rouge1 / cnt
        rouge2 = rouge2 / cnt
        rougeLsum = rougeLsum / cnt
        print("ranking rouge1: %.6f, rouge2: %.6f, rougeL: %.6f"%(rouge1, rouge2, rougeLsum))

    sent_length_score_check(output_path)
    sent_position_score_check(output_path)
    
    return {"rouge1":rouge1,"rouge2":rouge2,"rougeLsum":rougeLsum}

        

def validation(args, val_check_file_path, val_count, val_dataloader, model, tok):
    
    # Model
    model.eval()
    _model = model
    
    # Output 
    model_name=args.result_path
    output_path = "/home/nlplab/hdd1/yoo/BRIO/data/cnndm/diverse/val_output/"+model_name+"/"+str(val_count)
    if os.path.exists(output_path):
        assert "save path exist"
    os.makedirs(output_path,exist_ok=True)
    
    # Metric
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    rouge1, rouge2, rougeLsum = 0, 0, 0
    cnt = 0
    
    mle_loss = 0
    total_acc = 0
    acc_cnt = 0
    total_soft_acc=0
    soft_acc_cnt=0
    
    if args.smooth > 0: #####
        mle_fn = label_smoothing_loss(ignore_index=tok.pad_token_id, epsilon=args.smooth) #####
    else: #####
        mle_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id) #####
    
    _model.scoring_mode() #####
    
    is_val=1
    # Validation
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            for b in batch:
                if b != "data" and b!="sent_lens" and b!="sent_rouges" and b!="idx" and batch[b] is not None:
                    batch[b] = batch[b].to(args.gpuid)
            samples = batch["data"]
            
            output = model(args, batch["src_input_ids"], batch["candidate_ids"], batch["decoder_sent_mask"])
            similarity, gold_similarity, token_level_score = output['score'], output['summary_score'], output['token_level_scores']
            
            # Ranking loss
            ValRankingLoss(args, output_path, output["div_num"], batch["idx"][0], batch["sent_lens"], token_level_score, similarity)            
            total_loss, sum_loss, sent_loss, acc, soft_acc  = RankingLoss(args, is_val,  output["div_num"], batch["sent_rouges"], batch["sent_lens"], token_level_score, similarity, gold_similarity)
            total_acc+=acc
            acc_cnt+=1
            total_soft_acc+=soft_acc
            soft_acc_cnt+=1
            
            similarity = similarity.cpu().numpy()
            max_ids = similarity.argmax(1)
            
            # mle loss
            probs = output["probs"]  
            probs = output["probs"][:, :-1] 
            gold = batch["candidate_ids"][:, 0, 1:] 
            mle_loss += mle_fn(probs.transpose(1, 2), gold)
            
            for j in range(similarity.shape[0]):
                sample = samples[j]
                sents = sample["candidates"][max_ids[j]][0]
                score = rouge_scorer.score("\n".join(sample["abstract"]), "\n".join(sents))
                rouge1 += score["rouge1"].fmeasure
                rouge2 += score["rouge2"].fmeasure
                rougeLsum += score["rougeLsum"].fmeasure
                cnt += 1
                
    rouge1 = rouge1 / cnt
    rouge2 = rouge2 / cnt
    rougeLsum = rougeLsum / cnt
    mle_loss = mle_loss / cnt
    
    sent_length_score_check(output_path,val_check_file_path) #####
    sent_position_score_check(output_path,val_check_file_path) #####
    
    model.train()
    return {
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeLsum": rougeLsum,
        "ranking_loss": sum_loss, #####
        "mle_loss": mle_loss,
        "sent_loss": sent_loss,
        "accuracy": total_acc/acc_cnt,
        "raw_sent_loss": total_soft_acc/cnt,
        } 


def train(args):
    # Config
    if args.dataset == "cnndm":
        cnndm_setting(args)
    
    # Initialize
    torch.manual_seed(args.seed) #####
    torch.cuda.manual_seed_all(args.seed) #####
    np.random.seed(args.seed) #####
    random.seed(args.seed) #####
    gpuid = args.gpuid
    id = len(os.listdir("./checkpoints")) 
    date = datetime.now().strftime("%y-%m-%d")
    save_path = f"./checkpoints/{date}_{args.result_path}"
    
    wandb.init(
        project="Abstractive Summarization",
        # project="Abstractive Summarization tmp",
        name=args.result_path,
        config=args,
    )
    wandb.define_metric("val_ranking_loss", summary="min")
    wandb.define_metric("val_ranking_rouge1", summary="max")
    wandb.define_metric("val_ranking_rouge2", summary="max")
    wandb.define_metric("val_ranking_rougeLsum", summary="max")
    
    # Checkpoint
    if os.path.exists(save_path) and not ("test" in save_path):
        assert "save path exist"
    os.makedirs(save_path,exist_ok=True)

    # Output...
    val_check_path = "/home/nlplab/hdd1/yoo/BRIO/data/cnndm/diverse/val_output/"+args.result_path
    if os.path.exists(val_check_path):
        assert "save path exist"
    os.makedirs(val_check_path,exist_ok=True)
    val_check_file_path= val_check_path+'/'+"val_check.txt"
    
    # Dataset
    if args.is_pegasus:
        tok = PegasusTokenizer.from_pretrained(args.model_type)
    else:
        tok = BartTokenizer.from_pretrained(args.model_type)
        
    train_set = BrioDataset(args.datapath+"/new_train", tok, max_len=args.max_len, total_len=args.total_len, is_test=False, is_sorted=True, is_pegasus=args.is_pegasus)
    val_set = BrioDataset(args.datapath+"/new_val", tok, max_len=512, total_len=args.total_len, is_test=True, is_sorted=False, is_pegasus=args.is_pegasus)

    collate_fn = partial(collate_mp_brio, pad_token_id=tok.pad_token_id,  args=args, is_test=False)
    collate_fn_val = partial(collate_mp_brio, pad_token_id=tok.pad_token_id,  args=args, is_test=True)
    
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_val)
        
    # Model
    model = BRIO(args, tok.pad_token_id)
    if len(args.model_pt) > 0: 
        model.load_state_dict(torch.load(os.path.join("./checkpoints", args.model_pt), map_location=f'cuda:{gpuid}')) 
    model = model.cuda()
    model.train()
    
    model.scoring_mode() #####
    
    # Loss
    if args.smooth > 0: 
        mle_fn = label_smoothing_loss(ignore_index=tok.pad_token_id, epsilon=args.smooth) 
    else: #####
        mle_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id) #####
        
    optimizer = optim.Adam(model.parameters())
    lr=args.lr
    
    # Evaluation function
    if args.dataset == "xsum": #####
        def eval_fn(rouge1, rouge2, rougeLsum): #####
            return 1 - 2 * rouge1 * rouge2 / (rouge1 + rouge2) #####
    else: 
        def eval_fn(rouge1, rouge2, rougeLsum): 
            return 1 - (rouge1 * rouge2 + rougeLsum) / 3 
        
    all_step_cnt = 0 # 초기화 x
    maximum_accuracy = 0
    # maximum_soft_accuracy = 0
    minimum_ranking_loss = 100
    # minimum_mle_loss = 1e5
    
    is_val=0
    val_count=0
    # Training
    for epoch in range(args.epoch):
        optimizer.zero_grad() 
        
        step_cnt = 0 # 1~8
        epoch_step = 0 # epoch이 시작할 때 초기화 (한 accumulate step마다+1)
        
        avg_ranking_loss = 0
        avg_mle_loss = 0
        avg_loss = 0
        avg_sum_loss = 0 
        avg_sent_loss = 0 
        avg_acc=0
        avg_soft_acc=0
        
        for (i, batch) in enumerate(tqdm(train_dataloader)):     
            for b in batch:
                if b != "data" and b!="sent_lens" and b!="sent_rouges" and b!="idx" and batch[b] is not None:
                    batch[b] = batch[b].to(gpuid)
            step_cnt += 1

            # forward pass
            output = model(args, batch["src_input_ids"], batch["candidate_ids"], batch["decoder_sent_mask"])
            similarity, gold_similarity, token_level_score = output['score'], output['summary_score'], output['token_level_scores'] 
            
            # Ranking loss
            ranking_loss, sum_loss, sent_loss, acc, soft_acc  = RankingLoss(args, is_val, output["div_num"], batch["sent_rouges"], batch["sent_lens"], token_level_score, similarity, gold_similarity, batch["idx"])
            # mle loss
            probs = output["probs"]  # [batch_size, seq_len, word_num]
            probs = output["probs"][:, :-1]  # truncate last token (eos token?) [batch_size, seq_len-1, word_num]
            gold = batch["candidate_ids"][:, 0, 1:]  # shift right (sos token?) [batch_size, seq_len] -> [batch_size, seq_len-1]
            mle_loss = mle_fn(probs.transpose(1, 2), gold)
            # Total loss
            loss = args.rank_weight * ranking_loss + args.mle_weight * mle_loss
            loss = loss / args.accumulate_step
            
            avg_loss += loss.item() 
            avg_mle_loss += mle_loss.item() / args.accumulate_step
            avg_ranking_loss += ranking_loss.item() / args.accumulate_step
            avg_sum_loss += sum_loss.item() / args.accumulate_step
            avg_sent_loss += sent_loss.item() / args.accumulate_step
            avg_acc += acc / args.accumulate_step
            avg_soft_acc += soft_acc / args.accumulate_step

            loss.backward() # gradient calculation
            
            # Update at every accumulate step
            if step_cnt == args.accumulate_step:
                step_cnt = 0
                epoch_step += 1
                all_step_cnt += 1
                if args.fixed_lr:
                    if all_step_cnt % (args.eval_interval*args.lr_interval) == 0 and all_step_cnt != 0 and step_cnt == 0:
                        lr=lr*0.5
                else:
                    lr = args.max_lr * min(all_step_cnt ** (-0.5), all_step_cnt * (args.warmup_steps ** (-1.5)))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                optimizer.step() # gradient update
                optimizer.zero_grad() #####
                
            # Report at every report freq
            if epoch_step % args.report_freq == 0 and step_cnt == 0:
                print("id: %d"%id)
                print("epoch: %d, batch: %d, avg loss: %.6f, avg ranking loss: %.6f, avg mle loss: %.6f"
                %(epoch+1, epoch_step, avg_loss / args.report_freq, avg_ranking_loss / args.report_freq, avg_mle_loss / args.report_freq))
                log = {
                    "epoch": epoch+1,
                    "batch": epoch_step,
                    "current_lr": lr, 
                    "avg_loss": avg_sum_loss / args.report_freq,
                    "avg_ranking_loss": avg_sum_loss / args.report_freq,
                    "avg_mle_loss": avg_mle_loss / args.report_freq,
                    "avg_sum_loss": avg_sum_loss / args.report_freq,
                    "avg_sent_loss": avg_sent_loss / args.report_freq,
                    "avg_acc": avg_acc / args.report_freq,
                    "avg_raw_sent_loss": avg_soft_acc / args.report_freq,
                    "step": all_step_cnt,
                }
                wandb.log(log)
                avg_mle_loss, avg_ranking_loss, avg_loss, avg_sum_loss, avg_sent_loss, avg_acc, avg_soft_acc= 0, 0, 0, 0, 0, 0, 0
            del similarity, gold_similarity, loss, mle_loss, ranking_loss, sum_loss, sent_loss, output, probs

            # Validate at every eval interval
            if all_step_cnt % args.eval_interval == 0 and all_step_cnt != 0 and step_cnt == 0:
            #if True:
                val_dict={} 

                result = validation(args, val_check_file_path, val_count, val_dataloader, model, tok)
                loss = eval_fn(result["rouge1"], result["rouge2"], result["rougeLsum"])
                # sent_loss=result["sent_loss"] 
                accuracy=result["accuracy"]
                # soft_accuracy=result["soft_accuracy"]
                
                # if accuracy > maximum_accuracy:
                #     maximum_accuracy = accuracy
                #     torch.save(model.state_dict(), os.path.join(save_path, "model_acc.bin"))
                #     val_dict["best_acc_epoch"]=epoch
                #     val_dict["best_acc_batch"]=i / args.accumulate_step

                # if soft_accuracy > maximum_soft_accuracy:
                #     maximum_soft_accuracy = soft_accuracy
                #     torch.save(model.state_dict(), os.path.join(save_path, "model_soft_acc.bin"))
                #     val_dict["best_soft_acc_epoch"]=epoch
                #     val_dict["best_soft_acc_batch"]=i / args.accumulate_step


                if loss < minimum_ranking_loss :
                    minimum_ranking_loss = loss
                    torch.save(model.state_dict(), os.path.join(save_path, "model_best.bin"))
                    val_dict["best_ranking_epoch"]=epoch 
                    val_dict["best_ranking_batch"]=i / args.accumulate_step 

                model_name="model_"+str(epoch_step)+".bin"
                torch.save(model.state_dict(), os.path.join(save_path, model_name))
                
                print("val ranking loss: %.6f"%(loss))
                print("val ranking rouge1: %.6f, rouge2: %.6f, rougeLsum: %.6f"
                %(result["rouge1"], result["rouge2"], result["rougeLsum"]))
                
                val_dict["val_ranking_loss"]=loss
                val_dict["val_ranking_rouge1"]=result["rouge1"]
                val_dict["val_ranking_rouge2"]=result["rouge2"]
                val_dict["val_ranking_rougeLsum"]=result["rougeLsum"]
                val_dict["val_sent_loss"]=result["sent_loss"]
                val_dict["accuracy"]=result["accuracy"]
                # val_dict["soft_accuracy"]=result["soft_accuracy"]
                
                torch.save(model.state_dict(), os.path.join(save_path, "model_current.bin"))
                torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.bin"))
                
                log.update(val_dict)
                wandb.log(log)
                val_count+=1


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--gpuid", type=int, help="gpu id")
    parser.add_argument("-e", "--evaluate", action="store_true", help="evaluation")
    parser.add_argument("--dataset", default="cnndm", type=str, help="dataset") 
    parser.add_argument("--model_pt", default="", type=str, help="model path")
    parser.add_argument("--result_path", default="test", type=str, help="model path")
    parser.add_argument("--epoch", default=3, type=int)
    
    parser.add_argument("--length_penalty", default=2.0, type=float)
    parser.add_argument("--mle_weight", default=0, type=float)
    parser.add_argument("--margin", default=0.001, type=float)
    parser.add_argument("--gold_weight", default=0, type=float)
    parser.add_argument("--gold_margin", default=0, type=float)
    
    parser.add_argument("--fixed_lr", action="store_false") 
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--max_lr", default=2e-3, type=float)
    parser.add_argument("--lr_interval", default=20, type=float)
    parser.add_argument("--warmup_steps", default=10000, type=int)
    parser.add_argument("--small_dataset", default=False, action="store_true")
    
    parser.add_argument("--sent_loss", default=False, action="store_true")
    parser.add_argument("--sent_loss_weight", default=0.02, type=float)
    parser.add_argument("--sent_margin", default=0.02, type=float)
    parser.add_argument("--sent_length_penalty", default=1.0, type=float)
        
    parser.add_argument("--inter_sent_loss_weight", default=0.005, type=float)
    parser.add_argument("--inter_sent_margin", default=0.02, type=float)
    parser.add_argument("--random_sent_num", default=3, type=float)
    
    parser.add_argument("--decoder_sent_mask", default=False, action="store_true")
    parser.add_argument("--layer_apply_sent_mask", default=0, type=int)
    
    parser.add_argument("--inter", default=False, action="store_true")
    
    parser.add_argument("--softmax", default=False, action="store_true")
    
    
    args = parser.parse_args()
    
    if args.evaluate:
        with torch.cuda.device(args.gpuid): 
            evaluation(args)
    else:
        with torch.cuda.device(args.gpuid): 
            train(args)
        