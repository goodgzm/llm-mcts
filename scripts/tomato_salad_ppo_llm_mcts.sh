CUDA_VISIBLE_DEVICES=1 \
python twosome-mcts/overcooked/mcts_policy_pomdp_inference.py \
  --exp-name "tomato_salad_llm" \
  --num-envs 1 \
  --num-steps 64 \
  --task 0 \
  --env-id "Overcooked-LLMA-v4" \
  --record-path "workdir/overcooked/mcts" \
  --normalization-mode "word" \
  --env-reward 0.2 1 0.1 0.001 \
  --stochastic 0 \
  --value-weight 0.5 \
  --path-num 2000 \
  --depth 20 \
  --llm-base-model "meta-llama/Llama-3.1-8B-Instruct" \
  --llm-base-model-path "/data/gzm/TWOSOME-main/model/meta-llama/Llama-3.1-8B-Instruct" \


