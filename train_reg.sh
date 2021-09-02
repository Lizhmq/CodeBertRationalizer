export CUDA_VISIBLE_DEVICES=1
LANG=PYTHON
# DATADIR=../bigJava/datasets
DATADIR=../great/
# DATADIR=../CuBert/wrong_op
# OUTPUTDIR=./save/java-new-head-1
OUTPUTDIR=./save/varmis-head-1-0.1
# OUTPUTDIR=./save/pyop-reg-1
PRETRAINDIR=microsoft/codebert-base
LOGFILE=varmis-reg-1-0.1.log
PER_NODE_GPU=1
LAMBDA=0.1

# 4e-5
# 1e-4 too big
# varmis 2 * 4 * 8, 2epoch
# wrong op 12 * 3 * 2
# java op 1 * 4 * 8
# calls 1 * 1 * 64
# batchsize 72 better than 24


# -m torch.distributed.launch --nproc_per_node=$PER_NODE_GPU
python  run_reg.py \
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
        --per_gpu_eval_batch_size=16 \
        --gradient_accumulation_steps=8 \
        --num_train_epochs=8 \
        --logging_steps=100 \
        --save_steps=16000 \
        --overwrite_output_dir \
        --seed=2233 \
        --reghead=-1 \
        --lambd=$LAMBDA \
        --prob=0.1 \
        --mlm 
