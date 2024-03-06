MODEL_SAVE_DIR=save/

export HF_ENDPOINT=https://hf-mirror.com

mkdir -p $MODEL_SAVE_DIR

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python \
    bitdelta/train.py \
    --base_model /home/wanghanqing/projects/models/model_ver2/Mixtral-8x7B-v0.1 \
    --finetuned_model /home/wanghanqing/projects/models/model_ver2/Mixtral-8x7B-v0.1 \
    --save_dir $MODEL_SAVE_DIR \
    --batch_size 1 \
    --num_steps 100 \
    --save_full_model True \
    --train \

    # --layers "layers.5."\
    # /data/public/opensource_models/WizardLM/WizardMath-7B-V1.0/
