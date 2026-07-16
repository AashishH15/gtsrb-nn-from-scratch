"""Numerical gradient check for the CNN layers (debug/verification)."""

import numpy as np

from model import TinyCNN, cross_entropy, softmax


def _loss_at(model, name, idx, delta, X, y):
    if name == "w":
        orig = model.w.copy()
        new = orig.flatten().copy()
        new[idx] += delta
        model.w = new.reshape(orig.shape)
        ll = model.forward(X)
        model.w = orig
    elif name == "b":
        orig = model.b.copy()
        new = orig.copy()
        new[idx] += delta
        model.b = new
        ll = model.forward(X)
        model.b = orig
    elif name == "conv_w":
        orig = model.conv.w.copy()
        new = orig.flatten().copy()
        new[idx] += delta
        model.conv.w = new.reshape(orig.shape)
        ll = model.forward(X)
        model.conv.w = orig
    elif name == "conv_b":
        orig = model.conv.b.copy()
        new = orig.copy()
        new[idx] += delta
        model.conv.b = new
        ll = model.forward(X)
        model.conv.b = orig
    return cross_entropy(ll, y)


def main() -> None:
    np.random.seed(0)
    m = TinyCNN(num_classes=3, seed=1)
    X = np.random.rand(3, 32, 32, 3).astype(np.float32)
    y = np.array([0, 1, 2])

    m.forward(X)
    grads = m.backward(y)

    rng = np.random.default_rng(3)
    max_diff = 0.0
    for name in ["w", "b", "conv_w", "conv_b"]:
        if name == "conv_w":
            flat = m.conv.w
        elif name == "conv_b":
            flat = m.conv.b
        else:
            flat = getattr(m, name)
        idxs = rng.integers(0, flat.size, size=2)
        for i in idxs:
            eps = 1e-4
            num = (_loss_at(m, name, int(i), eps, X, y) - _loss_at(m, name, int(i), -eps, X, y)) / (2 * eps)
            ana = grads[name].flat[int(i)]
            d = abs(num - ana)
            max_diff = max(max_diff, d)
            print(f"{name}[{int(i)}]  num={num:+.5f}  ana={ana:+.5f}  diff={d:.2e}")
    print(f"\nMAX DIFF: {max_diff:.2e}")
    print("PASS" if max_diff < 1e-5 else "FAIL")


if __name__ == "__main__":
    main()
