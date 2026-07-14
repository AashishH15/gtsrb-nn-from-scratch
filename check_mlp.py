"""Sanity check for MLP forward pass and loss (commit 3)."""

import numpy as np

from model import MLP, cross_entropy, accuracy


def main() -> None:
    num_classes = 5
    mlp = MLP(num_classes=num_classes, seed=1)

    print(f"Total params: {mlp.count_params():,}")

    # Tiny synthetic batch: 8 images, 32x32x3
    X = np.random.rand(8, 32, 32, 3).astype(np.float32)
    y = np.random.randint(0, num_classes, size=8)

    logits = mlp.forward(X)
    print(f"Logits shape: {logits.shape}  (expected (8, {num_classes}))")

    loss = cross_entropy(logits, y)
    acc = accuracy(logits, y)
    print(f"Loss: {loss:.4f}  (random init should be near ln(5)={np.log(num_classes):.4f})")
    print(f"Accuracy: {acc:.3f}  (random should be near {1 / num_classes:.3f})")

    # Verify softmax is a valid distribution
    proba = mlp.predict_proba(X)
    print(f"softmax row sums: min={proba.sum(axis=1).min():.4f} max={proba.sum(axis=1).max():.4f}")

    assert logits.shape == (8, num_classes), "logits shape wrong"
    assert np.isfinite(loss), "loss not finite"
    assert abs(proba.sum(axis=1).min() - 1.0) < 1e-5, "softmax rows must sum to 1"
    assert abs(proba.sum(axis=1).max() - 1.0) < 1e-5, "softmax rows must sum to 1"
    print("All checks passed.")


if __name__ == "__main__":
    main()
