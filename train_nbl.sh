export CUDA_VISIBLE_DEVICES=1           ###
DATADIR=../c-treesitter/nbl          ###
OUTPUTDIR=./save/nbl2           ###
PRETRAINDIR=microsoft/codebert-base
LOGFILE=nbl2.log                        ### and train_name
PER_NODE_GPU=1
PER_GPU_BATCH_TRAIN=12
PER_GPU_BATCH_EVAL=24
GRAD_ACC=4
EPOCH=10
BLOCKSIZE=512


python  run_nbl.py \
        --data_dir=$DATADIR \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --block_size=$BLOCKSIZE \
        --do_train \
        --node_index 0 \
        --gpu_per_node $PER_NODE_GPU \
        --learning_rate=4e-5 \
        --weight_decay=0.01 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=$PER_GPU_BATCH_TRAIN \
        --per_gpu_eval_batch_size=$PER_GPU_BATCH_EVAL \
        --gradient_accumulation_steps=$GRAD_ACC \
        --num_train_epochs=$EPOCH \
        --logging_steps=200 \
        --save_steps=4000 \
        --overwrite_output_dir \
        --seed=2233 \
        --train_name train.json \
        --valid_name valid.json \
        --test_name test.json