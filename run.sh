MODEL_SAVE_DIR=save/calibrated_model_rank_bit_13b
diff_dir=save/
mkdir -p $MODEL_SAVE_DIR

CUDA_VISIBLE_DEVICES=0,1 python \
    bitdelta/train.py \
    --base_model /data/public/opensource_models/meta-llama/Llama-2-13b-hf/ \
    --finetuned_model  /data/public/opensource_models/meta-llama/Llama-2-13b-chat-hf/ \
    --save_dir $MODEL_SAVE_DIR \
    --batch_size 4 \
    --num_steps 200 \
    --save_full_model True \
    --save_diff_dir $diff_dir \
    --train

    # --layers "layers.5."\
    # /data/public/opensource_models/meta-llama/Llama-2-7b-chat-hf/

