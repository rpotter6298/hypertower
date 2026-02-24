# tower_watcher.py
import matplotlib.pyplot as plt

# tower_watcher.py
import matplotlib.pyplot as plt


class TowerWatcher:
    """
    Live monitor:
    - Cumulative batch-level: loss & accuracy per batch across all epochs.
    - Epoch batch-level: loss & accuracy per batch within the current epoch (resets each epoch).
    - TP/FP/TN/FN bar charts per tower, one chart each, new group each epoch.
    """

    def __init__(self):
        plt.ion()
        # 2 line plots (cum loss, cum acc), 2 line plots (epoch loss, epoch acc), 3 bar plots
        self.fig, self.axs = plt.subplots(7, 1, figsize=(10, 28))
        self.reset()

    def reset(self):
        # Cumulative batch-level
        self.global_batches = []
        self.loss_cum = {"fusion": [], "image": [], "meta": []}
        self.acc_cum = {"fusion": [], "image": [], "meta": []}
        # Epoch batch-level
        self.epoch_batches = []
        self.loss_epoch_batch = {"fusion": [], "image": [], "meta": []}
        self.acc_epoch_batch = {"fusion": [], "image": [], "meta": []}
        # Epoch markers for cum plots
        self.epoch_markers = []
        # Stats per epoch for bars
        self.epoch_stats = {"fusion": [], "image": [], "meta": []}
        # Track current epoch
        self.current_epoch = -1

    def on_epoch_start(self, epoch):
        # mark epoch boundary in cumulative
        x = self.global_batches[-1] + 1 if self.global_batches else 0
        self.epoch_markers.append(x)
        # reset epoch batch-level data
        self.epoch_batches = []
        for d in [self.loss_epoch_batch, self.acc_epoch_batch]:
            for k in d:
                d[k].clear()
        self.current_epoch = epoch

    def on_batch_end(self, idx, stats: dict):
        # Cumulative
        self.global_batches.append(len(self.global_batches) + 1)
        for key, lk, ak in [
            ("fusion", "loss_f", "acc_f"),
            ("image", "loss_i", "acc_i"),
            ("meta", "loss_m", "acc_m"),
        ]:
            self.loss_cum[key].append(stats.get(lk, 0))
            self.acc_cum[key].append(stats.get(ak, 0))
        # Epoch-level
        self.epoch_batches.append(len(self.epoch_batches) + 1)
        for key, lk, ak in [
            ("fusion", "loss_f", "acc_f"),
            ("image", "loss_i", "acc_i"),
            ("meta", "loss_m", "acc_m"),
        ]:
            self.loss_epoch_batch[key].append(stats.get(lk, 0))
            self.acc_epoch_batch[key].append(stats.get(ak, 0))
        # redraw
        self._draw_batch_plots()

    def on_epoch_end(self, epoch, stats: dict):
        # record per-epoch TP/FP/TN/FN
        for key in ["fusion", "image", "meta"]:
            self.epoch_stats[key].append(
                {
                    "tp": stats.get("tp", 0),
                    "fp": stats.get("fp", 0),
                    "tn": stats.get("tn", 0),
                    "fn": stats.get("fn", 0),
                }
            )
        self._draw_epoch_bars()

    def _draw_batch_plots(self):
        # Cumulative Loss
        ax = self.axs[0]
        ax.clear()
        ax.plot(self.global_batches, self.loss_cum["fusion"], label="Fusion")
        ax.plot(self.global_batches, self.loss_cum["image"], label="Image Tower")
        ax.plot(self.global_batches, self.loss_cum["meta"], label="MD Tower")
        for x in self.epoch_markers:
            ax.axvline(x=x, color="gray", linestyle="--")
        ax.set_ylabel("Cumulative Loss")
        ax.legend()

        # Epoch Loss
        ax = self.axs[1]
        ax.clear()
        ax.plot(self.epoch_batches, self.loss_epoch_batch["fusion"], label="Fusion")
        ax.plot(self.epoch_batches, self.loss_epoch_batch["image"], label="Image Tower")
        ax.plot(self.epoch_batches, self.loss_epoch_batch["meta"], label="MD Tower")
        ax.set_ylabel(f"Epoch {self.current_epoch+1} Loss")
        ax.set_xlabel("Batch (Epoch)")
        ax.legend()

        # Cumulative Accuracy
        ax = self.axs[2]
        ax.clear()
        ax.plot(self.global_batches, self.acc_cum["fusion"], label="Fusion")
        ax.plot(self.global_batches, self.acc_cum["image"], label="Image Tower")
        ax.plot(self.global_batches, self.acc_cum["meta"], label="MD Tower")
        for x in self.epoch_markers:
            ax.axvline(x=x, color="gray", linestyle="--")
        ax.set_ylabel("Cumulative Accuracy")
        ax.legend()

        # Epoch Accuracy
        ax = self.axs[3]
        ax.clear()
        ax.plot(self.epoch_batches, self.acc_epoch_batch["fusion"], label="Fusion")
        ax.plot(self.epoch_batches, self.acc_epoch_batch["image"], label="Image Tower")
        ax.plot(self.epoch_batches, self.acc_epoch_batch["meta"], label="MD Tower")
        ax.set_ylabel(f"Epoch {self.current_epoch+1} Accuracy")
        ax.set_xlabel("Batch (Epoch)")
        ax.legend()

        plt.pause(0.01)

    def _draw_epoch_bars(self):
        # Bar charts per tower
        for i, key in enumerate(["fusion", "image", "meta"]):
            ax = self.axs[4 + i]
            ax.clear()
            data = self.epoch_stats[key]
            epochs = list(range(1, len(data) + 1))
            tp = [d["tp"] for d in data]
            fp = [d["fp"] for d in data]
            tn = [d["tn"] for d in data]
            fn = [d["fn"] for d in data]
            width = 0.2
            ax.bar([e - width for e in epochs], tp, width, label="TP")
            ax.bar(epochs, fp, width, label="FP")
            ax.bar([e + width for e in epochs], tn, width, label="TN")
            ax.bar([e + 2 * width for e in epochs], fn, width, label="FN")
            ax.set_title(f"{key.title()} Tower Stats")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Count")
            ax.legend()
        plt.pause(0.01)
