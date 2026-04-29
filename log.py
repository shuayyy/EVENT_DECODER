from pathlib import Path
import os


class LiveTrainingPlotter:
    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.train_bit_accuracy = []
        self.val_bit_accuracy = []
        self.overfit_gap = []
        self.best_val_bit_accuracy = []
        self.enabled = False
        self.interactive = False
        self.plt = None
        self.figure = None
        self.axes = None
        self.lines = {}

        try:
            import matplotlib

            if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
                self.interactive = True
                matplotlib.use("TkAgg")
            else:
                matplotlib.use("Agg")

            import matplotlib.pyplot as plt

            self.plt = plt
            self.enabled = True
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.figure, self.axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
            self.figure.suptitle("Training Progress")

            self.lines["train_loss"], = self.axes[0].plot([], [], label="train_loss", color="#1f77b4")
            self.lines["val_loss"], = self.axes[0].plot([], [], label="val_loss", color="#ff7f0e")
            self.axes[0].set_ylabel("Loss")
            self.axes[0].grid(True, alpha=0.3)
            self.axes[0].legend()

            self.lines["train_bit_accuracy"], = self.axes[1].plot([], [], label="train_bit_accuracy", color="#2ca02c")
            self.lines["val_bit_accuracy"], = self.axes[1].plot([], [], label="val_bit_accuracy", color="#d62728")
            self.lines["best_val_bit_accuracy"], = self.axes[1].plot([], [], label="best_val_bit_accuracy", color="#9467bd", linestyle="--")
            self.axes[1].set_ylabel("Bit Accuracy")
            self.axes[1].set_ylim(0.0, 1.0)
            self.axes[1].grid(True, alpha=0.3)
            self.axes[1].legend()

            self.lines["overfit_gap"], = self.axes[2].plot([], [], label="overfit_gap", color="#8c564b")
            self.axes[2].axhline(0.0, color="black", linewidth=1, linestyle=":")
            self.axes[2].set_xlabel("Epoch")
            self.axes[2].set_ylabel("Train - Val")
            self.axes[2].grid(True, alpha=0.3)
            self.axes[2].legend()

            self.figure.tight_layout()

            if self.interactive:
                plt.ion()
                plt.show(block=False)
        except Exception:
            self.enabled = False

    def update(
        self,
        epoch,
        train_loss,
        val_loss,
        train_bit_accuracy,
        val_bit_accuracy,
        overfit_gap,
        best_val_bit_accuracy,
    ):
        if not self.enabled:
            return

        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_bit_accuracy.append(train_bit_accuracy)
        self.val_bit_accuracy.append(val_bit_accuracy)
        self.overfit_gap.append(overfit_gap)
        self.best_val_bit_accuracy.append(best_val_bit_accuracy)

        self.lines["train_loss"].set_data(self.epochs, self.train_loss)
        self.lines["val_loss"].set_data(self.epochs, self.val_loss)
        self.lines["train_bit_accuracy"].set_data(self.epochs, self.train_bit_accuracy)
        self.lines["val_bit_accuracy"].set_data(self.epochs, self.val_bit_accuracy)
        self.lines["best_val_bit_accuracy"].set_data(self.epochs, self.best_val_bit_accuracy)
        self.lines["overfit_gap"].set_data(self.epochs, self.overfit_gap)

        for axis in self.axes:
            axis.relim()
            axis.autoscale_view()

        self.axes[1].set_ylim(0.0, 1.0)
        self.axes[2].set_xlim(1, max(self.epochs))

        self.figure.tight_layout()
        self.figure.canvas.draw_idle()
        self.figure.savefig(self.output_path, dpi=150)

        if self.interactive:
            self.plt.pause(0.001)

    def close(self):
        if not self.enabled or not self.interactive:
            return
        self.plt.ioff()


class TrainingLogger:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.lines = []

    def log(self, message):
        print(message)
        self.lines.append(message)

    def save(self, print_message=True):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        if print_message:
            print(f"Saved training log to: {self.log_path}")


def format_epoch_log(
    epoch,
    train_loss,
    train_bit_accuracy,
    train_exact_accuracy,
    train_segment_metrics,
    val_metrics,
    task,
    best_val_bit_accuracy,
    best_epoch,
    overfit_gap,
):
    epoch_log = (
        f"Epoch {epoch}: "
        f"train_loss = {train_loss:.4f}, "
        f"train_bit_accuracy = {train_bit_accuracy:.4f}, "
        f"train_exact_accuracy = {train_exact_accuracy:.4f}, "
        f"train_data_accuracy = {train_segment_metrics['data_accuracy']:.4f}, "
        f"val_loss = {val_metrics['loss']:.4f}, "
        f"val_bit_accuracy = {val_metrics['bit_accuracy']:.4f}, "
        f"val_exact_accuracy = {val_metrics['exact_accuracy']:.4f}, "
        f"val_data_accuracy = {val_metrics['data_accuracy']:.4f}, "
        f"best_val_bit_accuracy = {best_val_bit_accuracy:.4f}, "
        f"best_epoch = {best_epoch}, "
        f"overfit_gap = {overfit_gap:.4f}"
    )
    if task == "ctc":
        epoch_log += (
            f", per_repetition_val_bit_accuracy = {val_metrics['bit_accuracy']:.4f}, "
            f"per_repetition_val_exact_accuracy = {val_metrics['exact_accuracy']:.4f}, "
            f"val_avg_pred_len = {val_metrics['avg_pred_length']:.2f}, "
            f"val_avg_target_len = {val_metrics['avg_target_length']:.2f}"
        )
        if "voted_chunk_bit_accuracy" in val_metrics:
            epoch_log += (
                f", voted_chunk_val_bit_accuracy = {val_metrics['voted_chunk_bit_accuracy']:.4f}, "
                f"voted_chunk_val_exact_accuracy = {val_metrics['voted_chunk_exact_accuracy']:.4f}"
            )
    return epoch_log
