from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import os
import scipy.ndimage
# 指定日志目录
log_dir = "/data/gzm/TWOSOME-main/workdir/origin/tomato_salad_ppo/Overcooked-LLMA-v4_task=0_exp_name=tomato_salad_ppo_seed=1_20250427_09_12_52/events.out.tfevents.1745759572.user-PR4910W.3479221.0"
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
    steps = [entry.step for entry in scalar_data]
    values = [entry.value for entry in scalar_data]
    print(f"the total number of values in '{target_tag}':", len(values))
    
    # 6. 对 value 做高斯平滑
    sigma = 2  # 控制平滑程度，sigma 越大越平滑
    smoothed_values = scipy.ndimage.gaussian_filter1d(values, sigma=sigma)

    # 7. 画图（不显示，直接保存）
    plt.figure(figsize=(10, 6))
    plt.plot(steps, smoothed_values, label=f"{target_tag} (smoothed)", color='blue')
    plt.xlabel('Step')
    plt.ylabel('episodic_return')
    plt.title(f"Training Curve: {target_tag} (Gaussian Smoothed)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 8. 保存到本地
    save_path = "policy_loss_smoothed.png"  # 保存的文件名
    plt.savefig(save_path)
    print(f"图已保存到：{os.path.abspath(save_path)}")

    plt.close()  # 关闭画布，不显示

    #     # 统计小于等于 0.3 的值
    count_greater_than_one = sum(1 for v in values[:] if v <= 0.3)
    index_greater_than_one = [i for i, v in enumerate(values) if v <= 0.3]
    print(f"Index of values <= 0.3 in '{target_tag}':", count_greater_than_one)
    # print(f"Index of values > 0.3 in '{target_tag}':", index_greater_than_one)

    # 统计大于 0.3 的值的个数
    count_greater_than_one = sum(1 for v in values[:] if v > 0.3)
    index_greater_than_one = [i for i, v in enumerate(values) if v > 0.3]
    print(f"Number of values > 0.3 in '{target_tag}':", count_greater_than_one)
    # print(f"Index of values > 0.3 in '{target_tag}':", index_greater_than_one)


#     # 统计大于 1 的值的个数
    count_greater_than_one = sum(1 for v in values[:] if v >= 0.5)
    index_greater_than_one = [i for i, v in enumerate(values) if v >= 0.5]
    print(f"Number of values >= 0.5 in '{target_tag}':", count_greater_than_one)
#     print(f"Index of values > 1 in '{target_tag}':", index_greater_than_one)

    count_greater_than_one = sum(1 for v in values[:] if v >= 0.8)
    index_greater_than_one = [i for i, v in enumerate(values) if v >= 0.8]
    print(f"Number of values >= 0.8 in '{target_tag}':", count_greater_than_one)
#     print(f"Index of values > 1 in '{target_tag}':", index_greater_than_one)

    count_greater_than_one = sum(1 for v in values[:] if v >= 0.9)
    index_greater_than_one = [i for i, v in enumerate(values) if v >= 0.9]
    print(f"Number of values >= 0.9 in '{target_tag}':", count_greater_than_one)
#     print(f"Index of values > 1 in '{target_tag}':", index_greater_than_one)

    count_greater_than_one = sum(1 for v in values[:] if v >= 1)
    index_greater_than_one = [i for i, v in enumerate(values) if v >= 1]
    print(f"Number of values >= 1 in '{target_tag}':", count_greater_than_one)
#     print(f"Index of values > 1 in '{target_tag}':", index_greater_than_one)

    
else:
    print(f"Tag '{target_tag}' not found in logs.")
