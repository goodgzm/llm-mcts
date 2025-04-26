CUDA_VISIBLE_DEVICES=1 \
python twosome-CotTot/virtualhome/cot_policy_pomdp_inference_v2.py \
  --exp-name "watch_tv_ppo_llm"\
  --num-envs 1 \
  --env-id "VirtualHome-v2" \
  --record-path "workdir/entertainment/cot" \
  --normalization-mode "word" \
  --stochastic 0 \
  --path-num 2000 \
  --depth 15 \
  --llm-base-model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  --llm-base-model-path "/data/gzm/TWOSOME-main/model/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


