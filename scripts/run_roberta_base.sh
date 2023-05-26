export RECLOR_DIR=reclor_data
export TASK_NAME=reclor
export MODEL_NAME=roberta-large

python run_multiple_choice_debug.py \
    --model_type roberta \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --evaluate_during_training \
    --do_test \
    --do_lower_case \
    --data_dir $RECLOR_DIR \
    --max_seq_length 384 \
    --svo_weight 0.5 \
    --per_gpu_eval_batch_size 1   \
    --per_gpu_train_batch_size 24   \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-05 \
    --num_train_epochs 15.0 \
    --output_dir Checkpoints/${TASK_NAME}_ori_/${MODEL_NAME} \
    --logging_steps 200 \
    --save_steps 200 \
    --adam_betas "(0.9, 0.98)" \
    --adam_epsilon 1e-6 \
    --no_clip_grad_norm \
    --warmup_proportion 0.1\
    --weight_decay 0.01 \
    --overwrite_output_dir