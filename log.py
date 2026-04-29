from pathlib import Path


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
