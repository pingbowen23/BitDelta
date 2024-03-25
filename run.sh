MODEL_SAVE_DIR=save/uncalibrated_mlp_0
Rank_dim=(128 185 32 64 128)
Bit_dim=(0 0 2464 1952 928)

mkdir -p $MODEL_SAVE_DIR
for i in {0..0}
do
    MODEL_SAVE_DIR="save/uncalibrated_attn"

    CUDA_VISIBLE_DEVICES=6,7 python \
        bitdelta/train2.py \
        --base_model /data/public/opensource_models/meta-llama/Llama-2-7b-hf/ \
        --finetuned_model /data/public/opensource_models/WizardLM/WizardMath-7B-V1.0/ \
        --save_dir $MODEL_SAVE_DIR \
        --batch_size 4 \
        --num_steps 200 \
        --save_full_model True \
        --rank_dim ${Rank_dim[i]} \
        --bit_dim ${Bit_dim[i]} 
done