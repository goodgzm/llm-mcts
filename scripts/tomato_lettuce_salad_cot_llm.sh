CUDA_VISIBLE_DEVICES=1 \
python twosome-CotTot/overcooked/cot_policy_pomdp_inference.py \
  --exp-name "tomato_lettuce_salad_llm" \
  --num-envs 1 \
  --task 3 \
  --env-id "Overcooked-LLMA-v3" \
  --record-path "workdir/overcooked/cot" \
  --normalization-mode "word" \
  --stochastic 0.2 \
  --path-num 2000 \
  --depth 15 \
  --llm-base-model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  --llm-base-model-path "/data/gzm/TWOSOME-main/model/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"