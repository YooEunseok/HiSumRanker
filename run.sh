# !/bin/bash

#===========================================================================================================================================================

# python -u main.py --gpuid 0 --result_path baseline_fixedlr_nomle 
# python -u main.py --gpuid 0 --result_path nomask_sent_nomle --sent_loss
# python -u main.py --gpuid 0 --result_path sent_inter_sw{0.02}_sm{0.2}_isw{0.005}_ism{0.2} --sent_loss
# python -u main.py --gpuid 0 --result_path sent_inter_sw{1}_sm{0.2}_nopen --sent_loss --sent_loss_weight 1 --sent_margin 0.2 --sent_length_penalty 1.0
# python -u main.py --gpuid 1 --result_path sent_inter_sw{0.5}_sm{0.2}_nopen --sent_loss --sent_loss_weight 0.5 --sent_margin 0.2 --sent_length_penalty 1.0
# python -u main.py --gpuid 0 --result_path sent_inter_sw{0.3}_sm{0.3}_nopen --sent_loss --sent_loss_weight 0.5 --sent_margin 0.3 --sent_length_penalty 2.0

#===========================================================================================================================================================

# python -u main.py --gpuid 1 -e --model_pt /home/nlplab/hdd1/yoo/BRIO/baseline_mask/checkpoints/github_baseline/model_ranking.bin 
# python -u main.py --gpuid 1 -e --model_pt /home/nlplab/hdd1/yoo/BRIO/baseline_mask/checkpoints/github_baseline/model_ranking.bin --length_penalty 1.0 --sent_length_penalty 1.0

# python -u main.py --gpuid 1 -e --model_pt /home/nlplab/hdd1/yoo/BRIO/baseline_mask/checkpoints/nomask_sent_nomle/model_ranking.bin
# python -u main.py --gpuid 1 -e --model_pt /home/nlplab/hdd1/yoo/BRIO/baseline_mask/checkpoints/nomask_sent_nomle/model_ranking.bin --sent_length_penalty 1.0

# python -u main.py --gpuid 1 -e --model_pt /home/nlplab/hdd1/yoo/BRIO/baseline_mask/checkpoints/23-10-07_sent_sentnopenlaty/model_ranking.bin --sent_length_penalty 1.0
# python -u main.py --gpuid 0 -e --model_pt /home/nlplab/hdd1/yoo/BRIO/baseline_mask/checkpoints/23-10-07_sent_sentnopenlaty/model_current.bin --sent_length_penalty 1.0

# python -u main.py --gpuid 0 -e --model_pt /home/nlplab/hdd1/yoo/BRIO/baseline_mask/checkpoints/23-10-02_mask_nomle/model_ranking.bin --decoder_sent_mask 
# python -u main.py --gpuid 0 -e --model_pt /home/nlplab/hdd1/yoo/BRIO/baseline_mask/checkpoints/23-10-02_mask_nomle/model_ranking.bin --decoder_sent_mask 

# python -u main.py --gpuid 0 -e --model_pt /home/nlplab/ssd1/yoo/BRIO/baseline_mask/checkpoints/23-09-30_mask_sent_fixedlr_nomle/model_ranking.bin

# python -u main.py --gpuid 0 -e --model_pt /home/nlplab/hdd1/yoo/BRIO/baseline_mask/checkpoints/23-10-08_sent_sw{0.02}_sm{0.03}/model_ranking.bin
# python -u main.py --gpuid 0 -e --model_pt /home/nlplab/hdd1/yoo/BRIO/baseline_mask/checkpoints/23-10-09_sent_sw{0.04}_sm{0.01}/model_ranking.bin
# python -u main.py --gpuid 0 -e --model_pt /home/nlplab/hdd1/yoo/BRIO/baseline_mask/checkpoints/23-10-14_sent_sw{0.003}_sm{0.3}_nopen/model_ranking.bin

# python -u main.py --gpuid 0 -e --model_pt /home/nlplab/hdd1/yoo/BRIO/baseline_mask/checkpoints/23-10-14_sent_sw{0.003}_sm{0.3}_nopen/model_ranking.bin
python -u main.py --gpuid 1 -e --model_pt /home/nlplab/hdd1/yoo/BRIO/baseline_mask/checkpoints/23-10-25_sent_sw{0.5}_sm{0.05}_intra_nopen/model_ranking.bin
