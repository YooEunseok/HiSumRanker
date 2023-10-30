import os
import json
from transformers import BartTokenizer


def sent_length_score_check(output_path, val_check_file_path=None):
    
    folder_path = output_path
    file_list = os.listdir(folder_path)
    file_count = len(file_list)
    print(file_count)

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    
    # 문장 길이에 따른 predicted score
    score_list=[0,0,0,0,0,0] 
    score_num=[0,0,0,0,0,0] 
    for n in range(file_count):
        with open(output_path+'/'+str(n)+'.json', 'r') as f:
            json_data = json.load(f)
            #print(n)
            for candidate in json_data["candidates_untok"]: 
                for i in range(len(candidate[0])):
                    score=candidate[3][i]
                    length=len(tokenizer(candidate[0][i])["input_ids"])
                    for j in range(len(score_num)):
                        if length>20*(len(score_num)-1-j):
                            score_list[len(score_num)-1-j]+=score 
                            score_num[len(score_num)-1-j]+=1
                            continue
    print(score_num)
    for i in range(len(score_list)):
        score_list[i]=round(score_list[i]/max(score_num[i],1),4)
    print("sent_length_score_check",score_list)
    
    if val_check_file_path!=None:
        with open(val_check_file_path,"w+") as f:
            f.write("sent_length_score_check: "+str(score_list)+"\n")
            f.close()

def sent_position_score_check(output_path, val_check_file_path=None):
    
    folder_path = output_path
    file_list = os.listdir(folder_path)
    file_count = len(file_list)
    print(file_count)
    
    #문장 위치(순번)에 따른 predicted score 
    score_list=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 
    score_num=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 
    for n in range(file_count):
        with open(output_path+'/'+str(n)+'.json', 'r') as f:
            json_data = json.load(f)
            #print(n)
            for candidate in json_data["candidates_untok"]: 
                for i in range(len(candidate[0])):
                    score=candidate[3][i]
                    #length=len(tokenizer(candidate[0][i])["input_ids"])
                    score_list[i]+=score #*length
                    score_num[i]+=1
                    continue
    print(score_num[:5])
    for i in range(len(score_list)):
        score_list[i]=round((score_list[i]/max(score_num[i],1)),4)
    print("sent_position_score_check",score_list[:5])
    
    if val_check_file_path!=None:
        with open(val_check_file_path,"a+") as f:
            f.write("sent_position_score_check: "+str(score_list[:5])+"\n")
            f.close()
        

                
#문장 길이에 따른 ROUGE score
# score_list=[0,0,0,0,0,0] 
# score_num=[0,0,0,0,0,0] 
# for n in range(file_count):
#     with open('/home/nlplab/ssd1/yoo/BRIO/data/cnndm/diverse/0911_2_new_test/'+str(n)+'.json', 'r') as f:
#         json_data = json.load(f)
#         print(n)
#         for candidate in json_data["candidates_untok"]: 
#             for i in range(len(candidate[0])):
#                 score=candidate[2][i]
#                 length=len(tokenizer(candidate[0][i])["input_ids"])
#                 for j in range(len(score_num)):
#                     if length>20*(len(score_num)-1-j):
#                         if score<2:
#                             score_list[len(score_num)-1-j]+=score
#                             score_num[len(score_num)-1-j]+=1
#                         continue
# print(score_num)
# for i in range(len(score_list)):
#     score_list[i]=score_list[i]/score_num[i]
# print(score_list)

# 문장 위치(순번)에 따른 ROUGE score
# score_list=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 
# score_num=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 
# for n in range(file_count):
#     with open('/home/nlplab/ssd1/yoo/BRIO/data/cnndm/diverse/new_test/'+str(n)+'.json', 'r') as f:
#         json_data = json.load(f)
#         print(n)
#         for candidate in json_data["candidates_untok"]: 
#             for i in range(len(candidate[0])): # 문장 개수 만큼 loop
#                 score=candidate[2][i]
#                 if score<2:
#                     score_list[i]+=score
#                     score_num[i]+=1
# print(score_num)
# for i in range(len(score_list)):
#     score_list[i]=score_list[i]/max(score_num[i],1)
# print(score_list)

