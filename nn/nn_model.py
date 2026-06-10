from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn


class NNModel:
    """MLP for price direction prediction (up / neutral / down)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_classes: int = 3,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.model: Optional[nn.Module] = None
        self.is_trained: bool = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Construct the network. No-op if already built."""
        if self.model is not None:
            return
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes),
        )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 0.001,
        epoch_callback: Optional[Callable[[int, dict], None]] = None,
    ) -> dict:
        """Train the model with an 80/20 time-ordered split."""
        self.build()

        n = len(X)
        split = int(n * 0.8)

        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        metrics: dict = {}

        for epoch in range(epochs):
            self.model.train()

            optimizer.zero_grad()
            logits_train = self.model(X_train_t)
            loss = criterion(logits_train, y_train_t)
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            train_acc = self._accuracy(logits_train, y_train_t)

            self.model.eval()
            with torch.no_grad():
                logits_val = self.model(X_val_t)
                val_loss = criterion(logits_val, y_val_t).item()
                val_acc = self._accuracy(logits_val, y_val_t)

            metrics = {
                "loss": train_loss,
                "accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }

            if epoch_callback is not None:
                epoch_callback(epoch, metrics)

        self.is_trained = True
        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(self, features: np.ndarray) -> np.ndarray:
        """Single-sample inference. Returns probabilities of shape (num_classes,)."""
        if not self.is_trained:
            raise RuntimeError("NNModel is not trained")

        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, input_size)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)  # (1, num_classes)
            probs = torch.softmax(logits, dim=1)
        return probs.squeeze(0).numpy()  # (num_classes,)

    def run_batch(self, features: np.ndarray) -> np.ndarray:
        """Batch inference. Returns probabilities of shape (N, num_classes)."""
        if not self.is_trained:
            raise RuntimeError("NNModel is not trained")

        x = torch.tensor(features, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
        return probs.numpy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str) -> None:
        """Save model weights to path."""
        if self.model is None:
            raise RuntimeError("NNModel has not been built yet")
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str) -> None:
        """Load model weights from path. Builds network first if needed."""
        self.build()
        state = torch.load(path, weights_only=True)
        self.model.load_state_dict(state)
        self.is_trained = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
        preds = logits.argmax(dim=1)
        return (preds == labels).float().mean().item()
