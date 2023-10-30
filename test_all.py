import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
from argparse import Namespace
import torch
from main import evaluation
import os

api = wandb.Api()
gpuid=0

runs = api.runs(path="sunnycloud56/Abstractive Summarization")

config_df=pd.DataFrame(columns=runs[0].config.keys())

model_dir="./checkpoints"

model_dir_list= os.listdir(model_dir)
print(model_dir_list)

result_df=pd.DataFrame(columns=["model_path","rouge1","rouge2","rougeLsum"])
if os.path.isfile("result_df.csv"):
    result_df=pd.read_csv("result_df.csv")
print(runs[0].config)

default_args = Namespace(**runs[0].config)
default_dict = {'lr': 1e-05, 'seed': 970903, 'epoch': 3, 'gpuid': 0, 'inter': False, 'adding': 0, 'margin': 0.001,
                'max_lr': 0.002, 'smooth': 0.1, 'dataset': 'cnndm', 'max_len': 120, 'no_gold': False, 'softmax': False,
                'datapath': '../data/cnndm/diverse', 'datatype': 'diverse', 'evaluate': False, 'fixed_lr': True,
                'model_pt': '', 'grad_norm': 0, 'normalize': True, 'num_beams': 4, 'sent_loss': True, 'total_len': 1024,
                'batch_size': 1, 'is_pegasus': False, 'mle_weight': 0, 'model_type': 'facebook/bart-large-cnn',
                'score_mode': 'log', 'gen_max_len': 140, 'gen_min_len': 55, 'gold_margin': 0, 'gold_weight': 0, 
                'lr_interval': 20, 'rank_weight': 10, 'report_freq': 100, 'result_path': 'sent_sw{2}_sm{}_m{}_intra',
                'sent_margin': 0.02, 'warmup_steps': 10000, 'eval_interval': 1000, 'small_dataset': False, 'length_penalty': 2,
                'accumulate_step': 8, 'random_sent_num': 3, 'sent_loss_weight': 2, 'decoder_sent_mask': False, 'inter_sent_margin': 0.02,
                'sent_length_penalty': 2, 'layer_apply_sent_mask': 0, 'inter_sent_loss_weight': 0.005}

for r in runs:
    print(r.name)
    try:
        print(max(r.history()["_step"]))
    except:
        continue
    if (max(r.history()["_step"])>200):
        arg_dict=default_dict
        arg_dict.update(r.config)
        args = Namespace(**arg_dict)
        print(args)
        match_dir=[]
        for i in model_dir_list:
            if i[9:]==args.result_path:
                match_dir.append(i)
        print(match_dir)
                
        for i in match_dir:
            print(i)
            print(result_df["model_path"].values)
            if i in result_df["model_path"].values:
                continue
            args.model_pt=os.path.join(i,"model_ranking.bin")
            args.gpuid=gpuid
            with torch.cuda.device(args.gpuid): 
                result=evaluation(args)
                result["model_path"]=i
                result_one_df = pd.DataFrame([result])
                result_df=pd.concat([result_df,result_one_df],ignore_index=True)
                result_df.to_csv("result_df.csv")
            
    
    df_the_dict = pd.DataFrame([r.config])
    config_df=pd.concat([config_df,df_the_dict],ignore_index=True)
    
    # config_df=config_df.append(r.config, ignore_index=True)
    
print(config_df)
config_df.to_csv("config_df.csv")