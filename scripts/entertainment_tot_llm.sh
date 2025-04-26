CUDA_VISIBLE_DEVICES=1 \
python twosome-CotTot/virtualhome/tot_policy_pomdp_inference_v2.py \
  --exp-name "watch_tv_ppo_llm"\
  --num-envs 1 \
  --env-id "VirtualHome-v2" \
  --record-path "workdir/entertainment/tot" \
  --normalization-mode "word" \
  --stochastic 0 \
  --path-num 2000 \
  --depth 15 \
  --llm-base-model "meta-llama/Llama-3.1-8B-Instruct" \
  --llm-base-model-path "/data/gzm/TWOSOME-main/model/meta-llama/Llama-3.1-8B-Instruct" \
  --expanded-num 2

