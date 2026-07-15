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
        self._flat = flat
        self._h1 = relu(flat @ self.w1 + self.b1)
        self._h2 = relu(self._h1 @ self.w2 + self.b2)
        return self._h2 @ self.w3 + self.b3

    def backward(self, y: np.ndarray) -> dict:
        """
        Backpropagation returning parameter gradients.

        Must be called after forward(). y : (N,) integer labels.
        Returns a dict of gradients for w1,b1,w2,b2,w3,b3.
        """
        n = y.shape[0]
        probs = softmax(self._h2 @ self.w3 + self.b3)

        # Gradient of cross-entropy w.r.t. logits: (probs - onehot) / N
        onehot = np.zeros_like(probs)
        onehot[np.arange(n), y] = 1.0
        d_logits = (probs - onehot) / n  # (N, C)

        d_w3 = self._h2.T @ d_logits
        d_b3 = d_logits.sum(axis=0)
        d_h2 = d_logits @ self.w3.T  # (N, hidden2)

        d_relu2 = d_h2 * (self._h2 > 0)
        d_w2 = self._h1.T @ d_relu2
        d_b2 = d_relu2.sum(axis=0)
        d_h1 = d_relu2 @ self.w2.T  # (N, hidden1)

        d_relu1 = d_h1 * (self._h1 > 0)
        d_w1 = self._flat.T @ d_relu1
        d_b1 = d_relu1.sum(axis=0)

        return {"w1": d_w1, "b1": d_b1, "w2": d_w2, "b2": d_b2, "w3": d_w3, "b3": d_b3}

    def update(self, grads: dict, lr: float, velocity: dict | None = None, momentum: float = 0.0) -> dict:
        """
        SGD step with optional momentum.

        velocity : running velocity dict (mutated in place, returned for reuse)
        momentum : 0.0 = plain SGD, e.g. 0.9 = classic momentum
        """
        if velocity is None:
            velocity = {k: np.zeros_like(v) for k, v in grads.items()}
        for k in grads:
            velocity[k] = momentum * velocity[k] - lr * grads[k]
            self.__dict__[k] += velocity[k]
        return velocity

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.forward(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X), axis=-1)

    def count_params(self) -> int:
        return sum(
            p.size for p in (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3)
        )


def _clip_grads(grads: dict, max_norm: float) -> dict:
    """Clip gradients to a global norm to keep SGD stable on tiny batches."""
    total = np.sqrt(sum(np.sum(g * g) for g in grads.values()))
    if total > max_norm:
        scale = max_norm / total
        return {k: v * scale for k, v in grads.items()}
    return grads


def train_mlp(
    model: MLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 20,
    lr: float = 0.1,
    batch_size: int = 128,
    momentum: float = 0.0,
    grad_clip: float = 5.0,
    seed: int = 0,
) -> dict:
    """
    Mini-batch SGD training loop. Returns a history dict with per-epoch
    train/val loss and accuracy for plotting.
    """
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    velocity = None

    for epoch in range(epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        epoch_correct = 0

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = X_train[idx]
            yb = y_train[idx]

            logits = model.forward(xb)
            grads = model.backward(yb)
            grads = _clip_grads(grads, grad_clip)
            velocity = model.update(grads, lr, velocity=velocity, momentum=momentum)

            epoch_loss += cross_entropy(logits, yb) * len(idx)
            epoch_correct += int(np.sum(np.argmax(logits, axis=-1) == yb))

        train_loss = epoch_loss / n
        train_acc = epoch_correct / n
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        if X_val is not None and y_val is not None:
            val_logits = model.forward(X_val)
            history["val_loss"].append(cross_entropy(val_logits, y_val))
            history["val_acc"].append(accuracy(val_logits, y_val))

        val_str = ""
        if X_val is not None and y_val is not None:
            val_str = f" | val_acc={history['val_acc'][-1]:.3f}"
        print(
            f"epoch {epoch + 1:>2}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}{val_str}"
        )

    return history
