CUDA_VISIBLE_DEVICES=1 \
python twosome-CotTot/overcooked/tot_policy_pomdp_inference.py \
  --exp-name "tomato_lettuce_salad_llm" \
  --num-envs 1 \
  --task 3 \
  --env-id "Overcooked-LLMA-v3" \
  --record-path "workdir/overcooked/tot" \
  --normalization-mode "word" \
  --stochastic 0 \
  --path-num 2000 \
  --depth 20 \
   --llm-base-model "meta-llama/Llama-3.1-8B-Instruct" \
  --llm-base-model-path "/data/gzm/TWOSOME-main/model/meta-llama/Llama-3.1-8B-Instruct" \
  --expanded-num 4