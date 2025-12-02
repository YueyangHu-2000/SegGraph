import matplotlib.pyplot as plt
import os

class TrainingMetricsPlotter:
    def __init__(self, save_dir="plots"):
        """
        初始化绘图器。

        Args:
            save_dir (str): 保存绘图的文件夹路径。
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_metrics(self, metrics, title="Training Metrics", save_name="metrics_plot.png"):
        """
        绘制训练过程中不同指标的曲线。

        Args:
            metrics (dict): 指标数据，键为指标类别（如"loss"、"acc"、"mIoU"），
                           值为字典，格式如下：
                {
                    "loss": {
                        "train": [0.9, 0.8, ...],
                        "val": [0.95, 0.85, ...]
                    },
                    "acc": {
                        "train": [0.7, 0.75, ...],
                        "val": [0.68, 0.72, ...]
                    }
                }
            title (str): 图像标题。
            save_name (str): 保存文件的名称。

        Raises:
            ValueError: 如果 metrics 格式不正确。
        """
        if not isinstance(metrics, dict):
            raise ValueError("`metrics` should be a dictionary.")

        num_subplots = len(metrics)
        if num_subplots == 0:
            raise ValueError("`metrics` should contain at least one category.")

        plt.figure(figsize=(12, 4 * num_subplots))

        for i, (metric_name, metric_data) in enumerate(metrics.items(), start=1):
            if not isinstance(metric_data, dict):
                raise ValueError(f"Values for {metric_name} should be a dictionary.")

            plt.subplot(num_subplots, 1, i)
            for subset_name, values in metric_data.items():
                if not isinstance(values, (list, tuple)):
                    raise ValueError(f"Values for {metric_name} - {subset_name} should be a list or tuple.")

                plt.plot(values, label=f"{subset_name} {metric_name}")

            plt.title(f"{metric_name.capitalize()} over Epochs", fontsize=14)
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel(metric_name.capitalize(), fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, linestyle="--", alpha=0.6)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save the plot
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print(f"Plot saved to {save_path}")

# 示例用法
if __name__ == "__main__":
    plotter = TrainingMetricsPlotter(save_dir="training_plots")

    # 模拟一些数据
    metrics_data = {
        "loss": {
            "train": [0.9, 0.8, 0.7, 0.6],
            "val": [0.95, 0.85, 0.75, 0.65]
        },
        "acc": {
            "train": [0.7, 0.75, 0.8, 0.85],
            "val": [0.68, 0.72, 0.77, 0.82]
        },
        "mIoU": {
            "train": [0.5, 0.55, 0.6, 0.65],
            "val": [0.48, 0.52, 0.58, 0.63]
        }
    }

    # 绘制并保存图像
    plotter.plot_metrics(metrics_data, title="Training Metrics Overview", save_name="metrics_overview.png")