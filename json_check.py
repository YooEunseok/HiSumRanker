import json
import os

with open(os.path.join("/home/nlplab/hdd1/yoo/BRIO/data/cnndm/diverse/new_train", "%d.json"%286147), "r") as f:
    data = json.load(f)
    
    # print(data)
    
with open('/home/nlplab/hdd1/yoo/BRIO/check.json', 'w', encoding='utf-8') as make_file:
    json.dump(data, make_file, indent="\t")