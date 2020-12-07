export CUDA_VISIBLE_DEVICES=5,6,7
LANG=JAVA
DATADIR=../bigJava/datasets
OUTPUTDIR=./save/java0/
# PRETRAINDIR=microsoft/codebert-base-mlm
PRETRAINDIR=../.code-bert-cache/codebert-base
LOGFILE=acc9.log
PER_NODE_GPU=3

# 4e-5
# 1e-4 too big

# batchsize 72 better than 24

python -m torch.distributed.launch --nproc_per_node=$PER_NODE_GPU run_classifier.py \
        --data_dir=$DATADIR \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=roberta \
        --block_size=512 \
        --do_train \
        --node_index 0 \
        --gpu_per_node $PER_NODE_GPU \
        --learning_rate=4e-5 \
        --weight_decay=0.01 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=8 \
        --per_gpu_eval_batch_size=10 \
        --gradient_accumulation_steps=3 \
        --num_train_epochs=20 \
        --logging_steps=100 \
        --save_steps=2000 \
        --overwrite_output_dir \
        --seed=2333 \
        --mlm 
