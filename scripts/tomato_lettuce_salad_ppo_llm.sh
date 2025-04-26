CUDA_VISIBLE_DEVICES=0 \
python twosome/overcooked/ppo_llm_pomdp.py \
  --exp-name "tomato_lettuce_salad_llm"\
  --policy-learning-rate 5e-7 \
  --value-learning-rate 1e-5 \
  --num-envs 2 \
  --num-steps 8 \
  --policy-num-minibatches 16 \
  --value-num-minibatches 4 \
  --update-epochs 1 \
  --total-timesteps 500000 \
  --critic-warm-up-steps 0 \
  --env-reward 0.2 1 0.1 0.001 \
  --target-kl 0.02 \
  --gradient-checkpointing-steps 16 \
  --task 3 \
  --env-id "Overcooked-LLMA-v3" \
  --record-path "workdir/origin" \
  --normalization-mode "word" \
  --llm-base-model "meta-llama/Llama-3.1-8B-Instruct" \
  --llm-base-model-path "/data/dengziwei/lcj_test_project/twosome/TWOSOME-main/hf_models/meta-llama/Llama-3.1-8B-Instruct"

