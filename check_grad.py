"""Numerical gradient check for MLP.backward (debug tool)."""

import numpy as np

from model import MLP, cross_entropy


def main() -> None:
    np.random.seed(0)
    m = MLP(num_classes=3, hidden1=5, hidden2=4, weight_scale=1.0, seed=1)
    X = np.random.randn(4, 32, 32, 3).astype(np.float32)
    y = np.array([0, 1, 2, 1])

    m.forward(X)
    grads = m.backward(y)

    def loss_at(name, idx, delta):
        orig = m.__dict__[name].copy()
        new = orig.flatten().copy()
        new[idx] += delta
        m.__dict__[name] = new.reshape(orig.shape)
        ll = m.forward(X)
        m.__dict__[name] = orig
        return cross_entropy(ll, y)

    rng = np.random.default_rng(7)
    max_diff = 0.0
    for name in ["w1", "b1", "w2", "b2", "w3", "b3"]:
        flat = m.__dict__[name]
        idxs = rng.integers(0, flat.size, size=3)
        for i in idxs:
            eps = 1e-4
            num = (loss_at(name, int(i), eps) - loss_at(name, int(i), -eps)) / (2 * eps)
            ana = grads[name].flat[int(i)]
            d = abs(num - ana)
            max_diff = max(max_diff, d)
            print(f"{name}[{int(i)}]  num={num:+.5f}  ana={ana:+.5f}  diff={d:.2e}")
    print(f"\nMAX DIFF: {max_diff:.2e}")
    print("PASS" if max_diff < 1e-6 else "FAIL")


if __name__ == "__main__":
    main()
