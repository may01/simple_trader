import os
import re
from pathlib import Path

from nn.nn_model import NNModel


class CheckpointManager:
    """Manages model checkpoints with versioning and best-model tracking."""

    def __init__(self, checkpoint_dir: str, model_name: str = "model") -> None:
        """
        Initialize CheckpointManager.

        Args:
            checkpoint_dir: Directory where checkpoints are stored
            model_name: Base name for checkpoint files (default: "model")

        Creates checkpoint_dir if it doesn't exist.
        """
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.best_metric = float("-inf")

        # Create directory if it doesn't exist
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def save(self, model: NNModel, metrics: dict, epoch: int) -> str:
        """
        Save model checkpoint atomically.

        Args:
            model: NNModel instance to save
            metrics: Dictionary containing validation metrics (must include 'val_accuracy')
            epoch: Epoch number for this checkpoint

        Returns:
            Path to the saved epoch checkpoint

        If val_accuracy improves, also saves a copy as _best.pt.
        Saves are atomic: write to .tmp then os.rename.
        """
        # Construct epoch checkpoint path
        epoch_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_epoch{epoch}.pt")
        tmp_path = epoch_path + ".tmp"

        # Atomically save to .tmp then rename
        try:
            model.save_model(tmp_path)
            os.rename(tmp_path, epoch_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        # Check if this is the best model so far
        val_accuracy = metrics["val_accuracy"]
        if val_accuracy > self.best_metric:
            self.best_metric = val_accuracy

            # Save best model atomically
            best_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pt")
            best_tmp_path = best_path + ".tmp"
            try:
                model.save_model(best_tmp_path)
                os.rename(best_tmp_path, best_path)
            except Exception:
                if os.path.exists(best_tmp_path):
                    os.remove(best_tmp_path)
                raise

        return epoch_path

    def load_best(self, model: NNModel) -> bool:
        """
        Load the best model checkpoint.

        Args:
            model: NNModel instance to load weights into

        Returns:
            True if loaded successfully, False if best checkpoint doesn't exist
        """
        best_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pt")
        if not os.path.isfile(best_path):
            return False

        model.load_model(best_path)
        return True

    def load_epoch(self, model: NNModel, epoch: int) -> bool:
        """
        Load a specific epoch checkpoint.

        Args:
            model: NNModel instance to load weights into
            epoch: Epoch number to load

        Returns:
            True if loaded successfully, False if checkpoint doesn't exist
        """
        epoch_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_epoch{epoch}.pt")
        if not os.path.isfile(epoch_path):
            return False

        model.load_model(epoch_path)
        return True

    def list_checkpoints(self) -> list[dict]:
        """
        List all epoch checkpoints.

        Returns:
            List of dicts with keys: epoch (int), path (str), size_bytes (int)
            Sorted by epoch number ascending
            Does NOT include _best.pt
        """
        checkpoints = []

        # Scan directory for epoch checkpoint files
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith(self.model_name) and filename.endswith(".pt"):
                # Skip best.pt files
                if "_best.pt" in filename:
                    continue

                # Extract epoch number from "{model_name}_epoch{epoch}.pt"
                if "_epoch" in filename:
                    try:
                        # Parse epoch number strictly: "{model_name}_epoch{N}.pt"
                        m = re.match(r"^.+_epoch(\d+)\.pt$", filename)
                        if m is None:
                            continue
                        epoch = int(m.group(1))

                        full_path = os.path.join(self.checkpoint_dir, filename)
                        size_bytes = os.path.getsize(full_path)

                        checkpoints.append(
                            {
                                "epoch": epoch,
                                "path": full_path,
                                "size_bytes": size_bytes,
                            }
                        )
                    except (ValueError, IndexError):
                        # Skip files that don't match expected format
                        pass

        # Sort by epoch number ascending
        checkpoints.sort(key=lambda x: x["epoch"])
        return checkpoints

    def cleanup(self, keep_best: bool = True, keep_last_n: int = 3) -> None:
        """
        Delete old epoch checkpoint files.

        Args:
            keep_best: If True, never delete _best.pt (default: True)
            keep_last_n: Number of most recent epoch files to keep (default: 3)
        """
        checkpoints = self.list_checkpoints()

        # Delete checkpoints older than keep_last_n
        if keep_last_n == 0:
            # Delete all epoch checkpoints
            to_delete = checkpoints
        else:
            # Delete all but the last keep_last_n
            to_delete = checkpoints[:-keep_last_n]

        for cp in to_delete:
            os.remove(cp["path"])

        # Optionally delete _best.pt
        if not keep_best:
            best_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pt")
            if os.path.isfile(best_path):
                os.remove(best_path)
