#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pw
export CUDA_VISIBLE_DEVICES=2


dataset="covost2"
model="tiny"
dataset_dir="path/to/covost2"
core_metric="bleu"
split="dev"
language="de"
logit_mask=${language}
vocab_cap=0.5
# for unconstraint gen (no vocab constaint), don't pass logit_mask
# for language zh and ar, we directly constrain vocab using script, so vocab_cap won't be needed
echo "currently testing ${model}"
exp_name="${language}_${model}_${split}"
python ../csasr_st.py \
--language ${language} \
--logit_mask ${logit_mask} \
--vocab_cap ${vocab_cap} \
--data_split ${split} \
--model ${model} \
--dataset ${dataset} \
--dataset_dir ${dataset_dir} \
--core_metric ${core_metric} \
--beam_size 5 \
--topk 1000 \
--task transcribe 
# >> "./logs/${dataset}/${exp_name}.log" 2>&1