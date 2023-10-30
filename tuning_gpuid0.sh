#/bin/bash

sent_weight=2

python -u main.py \
    --gpuid 0 \
    --result_path "sent_sw{$sent_weight}_sm{$sent_margin}_m{$margin}_intra" \
    --sent_loss \
    --sent_loss_weight $sent_weight \
    --length_penalty 2.0 \
    --sent_length_penalty 2.0 \
    --margin 0.001

# SENT_MARGIN=(0.01 0.02)
# SENT_WEIGHT=(0.04 0.03 0.02 0.015 0.01)

# for sent_margin in ${SENT_MARGIN[@]}; do
#     for sent_weight in ${SENT_WEIGHT[@]}; do
#         if [ $sent_weight == 0.025 ] && [ $sent_margin == 0.02 ]; then
#             continue
#         fi
#         python -u main.py \
#             --gpuid 0 \
#             --result_path "sent_sw{$sent_weight}_sm{$sent_margin}" \
#             --sent_loss \
#             --sent_loss_weight $sent_weight \
#             --sent_margin $sent_margin  
#     done
# done
