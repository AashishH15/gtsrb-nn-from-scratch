"""NumPy-only neural network models for GTSRB traffic sign classification."""

from __future__ import annotations

import numpy as np

from data import IMAGE_SIZE


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def cross_entropy(logits: np.ndarray, y: np.ndarray) -> float:
    """
    Average cross-entropy loss for integer class labels.

    logits : (N, C) pre-softmax scores
    y      : (N,) integer labels in [0, C)
    """
    probs = softmax(logits)
    n = y.shape[0]
    eps = 1e-12
    correct = probs[np.arange(n), y]
    return float(-np.mean(np.log(correct + eps)))


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    """Fraction of correct predictions."""
    preds = np.argmax(logits, axis=-1)
    return float(np.mean(preds == y))


class MLP:
    """
    Two-hidden-layer MLP on a flattened 32x32x3 image.

    Architecture: flatten(3072) -> 256 -> ReLU -> 128 -> ReLU -> C -> softmax
    This commit adds the forward pass and loss only; backprop + training
    are added in the next commit.
    """

    def __init__(
        self,
        num_classes: int = 5,
        hidden1: int = 256,
        hidden2: int = 128,
        weight_scale: float = 1e-2,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        in_dim = IMAGE_SIZE * IMAGE_SIZE * 3

        # He-ish scaling keeps initial activations in a sane range.
        self.w1 = rng.standard_normal((in_dim, hidden1)) * np.sqrt(2.0 / in_dim) * weight_scale
        self.b1 = np.zeros(hidden1)
        self.w2 = rng.standard_normal((hidden1, hidden2)) * np.sqrt(2.0 / hidden1) * weight_scale
        self.b2 = np.zeros(hidden2)
        self.w3 = rng.standard_normal((hidden2, num_classes)) * np.sqrt(2.0 / hidden2) * weight_scale
        self.b3 = np.zeros(num_classes)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass returning logits (pre-softmax scores).

        X : (N, 32, 32, 3) or (N, 3072)
        """
        flat = X.reshape(X.shape[0], -1).astype(np.float32)
        self._h1 = relu(flat @ self.w1 + self.b1)
        self._h2 = relu(self._h1 @ self.w2 + self.b2)
        return self._h2 @ self.w3 + self.b3

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.forward(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X), axis=-1)

    def count_params(self) -> int:
        return sum(
            p.size for p in (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3)
        )
