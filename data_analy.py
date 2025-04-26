from tensorboard.backend.event_processing import event_accumulator

# 指定日志目录
log_dir = ("/data/dengziwei/lcj_test_project/twosome/TWOSOME-main"
           "/workdir/mcts/tomato_salad_llm"
           "/Overcooked-LLMA-v4__task=tomato_salad_llm__stochastic=0.0__seed=1__20250227_14_59_41__llm=DeepSeek-R1-Distill-Llama-8B__normalization_mode=token__value_weight=0.5__path_num=2000__MCTS")
log_dir = "/data/dengziwei/lcj_test_project/twosome/TWOSOME-main/workdir/overcooked/mcts/tomato_salad_llm/Overcooked-LLMA-v4__task=tomato_salad_llm__stochastic=0.0__seed=1__20250303_10_33_47__llm=Llama-3.1-8B-Instruct__normalization_mode=token__value_weight=0.5__path_num=2000__MCTS"
# 加载日志数据
event_acc = event_accumulator.EventAccumulator(log_dir)
event_acc.Reload()

# 查看所有标量标签
scalar_tags = event_acc.Tags().get('scalars', [])
print("Available scalar tags:", scalar_tags)

# 提取特定标量的数据
target_tag = 'charts/episodic_return'  # 替换为你需要的标量名称
if target_tag in scalar_tags:
    scalar_data = event_acc.Scalars(target_tag)

    # 提取 value 列表
    values = [entry.value for entry in scalar_data]
    print(f"the total number of values in '{target_tag}':", len(values))
    # 统计大于 0.3 的值的个数
    count_greater_than_one = sum(1 for v in values[:] if v > 0.3)
    index_greater_than_one = [i for i, v in enumerate(values) if v > 0.3]
    print(f"Number of values > 0.3 in '{target_tag}':", count_greater_than_one)
    print(f"Index of values > 0.3 in '{target_tag}':", index_greater_than_one)

    # 统计小于于 0.3 的值
    index_greater_than_one = [i for i, v in enumerate(values) if v <= 0.3]
    print(f"Index of values <= 0.3 in '{target_tag}':", index_greater_than_one)

    # 统计大于 1 的值的个数
    count_greater_than_one = sum(1 for v in values[:] if v > 1)
    index_greater_than_one = [i for i, v in enumerate(values) if v > 1]
    print(f"Number of values > 1 in '{target_tag}':", count_greater_than_one)
    print(f"Index of values > 1 in '{target_tag}':", index_greater_than_one)

    # 统计小于于 0.3 的值
    index_greater_than_one = [i for i, v in enumerate(values) if v < 1]
    print(f"Index of values < 1 in '{target_tag}':", index_greater_than_one)

else:
    print(f"Tag '{target_tag}' not found in logs.")
