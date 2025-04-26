CUDA_VISIBLE_DEVICES=1 \
python twosome-mcts/overcooked/mcts_plusplus_policy_pomdp_inference.py \
  --exp-name "tomato_lettuce_salad_llm" \
  --num-envs 1 \
  --num-steps 64 \
  --task 3 \
  --env-id "Overcooked-LLMA-v3" \
  --record-path "workdir/mcts++" \
  --normalization-mode "word" \
  --env-reward 0.2 1 0.1 0.001 \
  --stochastic 0.2 \
  --value-weight 0.5 \
  --path-num 2000 \
  --depth 20 \
  --rnd False \
  --rnd-weight 0.5 \
  --transpositions False \
  --llm-base-model "meta-llama/Llama-3.1-8B-Instruct" \
  --llm-base-model-path "/data/dengziwei/lcj_test_project/twosome/TWOSOME-main/hf_models/meta-llama/Llama-3.1-8B-Instruct"


