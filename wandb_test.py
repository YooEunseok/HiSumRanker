# import wandb
import random

import torch
from torch import nn
import torch.nn.functional as F
# random.seed(1)
# wandb.init()
# # define a metric we are interested in the minimum of
# wandb.define_metric("loss", summary="min")
# # define a metric we are interested in the maximum of
# wandb.define_metric("acc", summary="max")
# for i in range(10):
#     log_dict = {
#         "loss": random.uniform(0, 1 / (i + 1)),
#         "acc": random.uniform(1 / (i + 1), 1),
#     }
#     wandb.log(log_dict)

token_score=[-1.1523,  -0.0497,  -0.0357,  -0.0881,  -1.7617,  -1.1642,  -0.0348,
            -0.0537,  -0.0598,  -0.0843,  -0.0284,  -0.1054,  -0.9269,  -1.2884,
            -0.1521,  -0.0885,  -0.0591,  -0.8034,  -1.6845,  -1.9529,  -0.1040,
            -0.1123,  -3.1970,  -2.6287,  -1.0839,  -0.0708,  -1.3479,  -0.1825,
            -0.1731,  -1.3818,  -3.2262,  -1.2059,  -0.2877,  -0.2154,  -0.9773,
            -0.2772,  -0.3906,  -1.8110,  -0.5270,  -0.1437,  -3.6799,  -1.0015,
            -0.2528,  -1.3800,  -3.2967,  -0.4848,  -0.3235,  -3.8878,  -3.3195,
            -2.7150,  -1.1373,  -1.1547,  -0.0513,  -6.9563,  -2.2570,  -3.0099,
            -1.5140,  -0.0975,  -0.2261,  -1.4492,  -5.8305,  -2.9880,  -7.0078,
            -0.9196,  -2.4521,  -1.4167,  -1.8579, -11.3560,  -4.4480,  -4.9089,
            -4.7647,  -3.0000,  -3.0018,  -3.1398 ]

print("summary score:", sum(token_score)/(len(token_score)**2))

sent_scores =[]
div_size = int(len(token_score) /6)
for i in range(6):
    a = token_score[i*div_size:(i+1)*div_size]
    sent_score = sum(a)/(len(a)**2)
    sent_scores.append(sent_score)

print(sent_scores)

sent_scores=torch.Tensor(sent_scores[::-1])
# sent_scores=torch.Tensor([random.uniform(-0.01,-0.03) for i in range(16)])

loss=0
for j in range(1,len(sent_scores)):
    pos_score=sent_scores[:-j]
    neg_score=sent_scores[j:]
    
    ones = torch.ones_like(pos_score)

    sent_loss_func = torch.nn.MarginRankingLoss(0.001*j) 
    loss+=sent_loss_func(pos_score,neg_score,ones)
    print(sent_loss_func(pos_score,neg_score,ones))

print(loss)
