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


def classification_report(logits: np.ndarray, y: np.ndarray, label_names: list[str]) -> str:
    """Per-class precision/recall/f1/accuracy, printed as a table."""
    preds = np.argmax(logits, axis=-1)
    lines = ["", "Per-class report", "-" * 60]
    header = f"{'class':<22}{'prec':>7}{'recall':>8}{'f1':>7}{'support':>9}"
    lines.append(header)
    overall_correct = 0
    for i, name in enumerate(label_names):
        tp = int(np.sum((preds == i) & (y == i)))
        fp = int(np.sum((preds == i) & (y != i)))
        fn = int(np.sum((preds != i) & (y == i)))
        support = int(np.sum(y == i))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        overall_correct += tp
        lines.append(
            f"{name:<22}{prec:>7.3f}{rec:>8.3f}{f1:>7.3f}{support:>9}"
        )
    lines.append("-" * 60)
    lines.append(f"accuracy: {overall_correct / len(y):.4f}  (n={len(y)})")
    return "\n".join(lines)


class Conv2D:
    """Single-stride conv with 'same' padding and KxK filters."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        # He scaling for the conv fan-in (kernel*kernel*in_ch)
        fan_in = kernel * kernel * in_ch
        scale = np.sqrt(2.0 / fan_in)
        self.w = rng.standard_normal((out_ch, in_ch, kernel, kernel)) * scale
        self.b = np.zeros(out_ch)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X : (N, H, W, C_in) -> (N, H, W, C_out)."""
        n, h, w, _ = X.shape
        k = self.w.shape[2]
        p = k // 2
        xp = np.pad(X, ((0, 0), (p, p), (p, p), (0, 0)), mode="constant")
        out = np.zeros((n, h, w, self.w.shape[0]), dtype=np.float32)
        # w shape: (Cout, Cin, k, k)
        for i in range(k):
            for j in range(k):
                window = xp[:, i : i + h, j : j + w, :]  # (N,H,W,Cin)
                kernel = self.w[:, :, i, j]  # (Cout, Cin)
                out += window @ kernel.T  # (N,H,W,Cout)
        out += self.b[None, None, None, :]
        return out

    def backward(self, dout: np.ndarray, X: np.ndarray) -> tuple:
        """Return (dX, grad_w, grad_b) given upstream gradient dout."""
        n, h, w, _ = X.shape
        k = self.w.shape[2]
        p = k // 2
        xp = np.pad(X, ((0, 0), (p, p), (p, p), (0, 0)), mode="constant")
        dXp = np.zeros_like(xp, dtype=np.float32)
        dw = np.zeros_like(self.w, dtype=np.float32)
        db = dout.sum(axis=(0, 1, 2))

        for i in range(k):
            for j in range(k):
                window = xp[:, i : i + h, j : j + w, :]  # (N,H,W,Cin)
                kernel = self.w[:, :, i, j]  # (Cout, Cin)
                # dw[c_out,c_in] = sum over N,H,W of window[n,h,w,c_in]*dout[n,h,w,c_out]
                dw[:, :, i, j] += dout.reshape(-1, dout.shape[-1]).T @ window.reshape(-1, window.shape[-1])
                # dXp patch = dout @ kernel
                dXp[:, i : i + h, j : j + w, :] += dout @ kernel
        dX = dXp[:, p : p + h, p : p + w, :]
        return dX, dw, db


class MaxPool2D:
    """2x2 max pooling with stride 2, no padding."""

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X : (N, H, W, C) with H,W even -> (N, H/2, W/2, C)."""
        n, h, w, c = X.shape
        h2, w2 = h // 2, w // 2
        out = np.zeros((n, h2, w2, c), dtype=np.float32)
        self._mask = np.zeros_like(X, dtype=bool)
        for i in range(h2):
            for j in range(w2):
                patch = X[:, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2, :]
                m = patch.max(axis=(1, 2), keepdims=True)
                out[:, i, j, :] = m[:, 0, 0, :]
                self._mask[:, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2, :] |= (
                    patch == m
                )
        return out

    def backward(self, dout: np.ndarray, X: np.ndarray) -> np.ndarray:
        n, h, w, c = X.shape
        h2, w2 = h // 2, w // 2
        dX = np.zeros_like(X, dtype=np.float32)
        for i in range(h2):
            for j in range(w2):
                dX[:, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2, :] += (
                    dout[:, i : i + 1, j : j + 1, :]
                    * self._mask[:, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2, :]
                )
        return dX


class TinyCNN:
    """
    Tiny 2-layer CNN:
    input 32x32x3
      -> Conv 3x3, 8 filters, 'same' -> ReLU -> 2x2 maxpool  (-> 16x16x8)
      -> flatten (2048) -> linear -> softmax (5)
    """

    def __init__(self, num_classes: int = 5, seed: int = 0) -> None:
        self.conv = Conv2D(in_ch=3, out_ch=8, kernel=3, seed=seed)
        self.pool = MaxPool2D()
        rng = np.random.default_rng(seed + 100)
        flat_dim = 16 * 16 * 8
        self.w = rng.standard_normal((flat_dim, num_classes)) * np.sqrt(2.0 / flat_dim)
        self.b = np.zeros(num_classes)

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32)
        self._input = X
        self._conv = relu(self.conv.forward(X))
        self._pooled = self.pool.forward(self._conv)
        flat = self._pooled.reshape(self._pooled.shape[0], -1)
        return flat @ self.w + self.b

    def backward(self, y: np.ndarray) -> dict:
        n = y.shape[0]
        logits = self._pooled.reshape(self._pooled.shape[0], -1) @ self.w + self.b
        probs = softmax(logits)
        onehot = np.zeros_like(probs)
        onehot[np.arange(n), y] = 1.0
        d_logits = (probs - onehot) / n  # (N, C)

        dw = self._pooled.reshape(n, -1).T @ d_logits
        db = d_logits.sum(axis=0)
        d_flat = d_logits @ self.w.T  # (N, 2048)
        d_pooled = d_flat.reshape(self._pooled.shape)

        d_conv = self.pool.backward(d_pooled, self._conv)
        d_conv_relu = d_conv * (self._conv > 0)
        _, dconv_w, dconv_b = self.conv.backward(d_conv_relu, self._input)
        return {
            "w": dw,
            "b": db,
            "conv_w": dconv_w,
            "conv_b": dconv_b,
        }

    def update(self, grads: dict, lr: float, velocity: dict | None = None, momentum: float = 0.0) -> dict:
        if velocity is None:
            velocity = {k: np.zeros_like(v) for k, v in grads.items()}
        for k in grads:
            velocity[k] = momentum * velocity[k] - lr * grads[k]
            if k == "conv_w":
                self.conv.w += velocity[k]
            elif k == "conv_b":
                self.conv.b += velocity[k]
            else:
                self.__dict__[k] += velocity[k]
        return velocity

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X), axis=-1)

    def count_params(self) -> int:
        return (
            self.conv.w.size
            + self.conv.b.size
            + self.w.size
            + self.b.size
        )


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
