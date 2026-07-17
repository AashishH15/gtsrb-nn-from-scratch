"""Train MLP and tiny CNN on the 5-class GTSRB subset and save comparison plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import load_prepared, summarize_split, CLASS_NAMES
from model import MLP, TinyCNN, train_mlp, accuracy, classification_report


def _plot(history: dict, title: str, out_path: Path) -> None:
    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(11, 4))
    ax_l.plot(history["train_loss"], label="train")
    if history.get("val_loss"):
        ax_l.plot(history["val_loss"], label="val")
    ax_l.set_title(f"{title} — loss")
    ax_l.set_xlabel("epoch")
    ax_l.set_ylabel("cross-entropy")
    ax_l.legend()

    ax_a.plot(history["train_acc"], label="train")
    if history.get("val_acc"):
        ax_a.plot(history["val_acc"], label="val")
    ax_a.set_title(f"{title} — accuracy")
    ax_a.set_xlabel("epoch")
    ax_a.set_ylabel("accuracy")
    ax_a.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train both models and save comparison plots.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    X_train, y_train, X_val, y_val, names = load_prepared()
    print(summarize_split(y_train, y_val, names))

    label_names = [CLASS_NAMES[c] for c in (0, 1, 14, 33, 38)]

    # --- MLP ---
    print("\nTraining MLP...")
    mlp = MLP(num_classes=5, seed=args.seed)
    mlp_hist = train_mlp(mlp, X_train, y_train, X_val, y_val, epochs=args.epochs, lr=0.1, batch_size=128)
    mlp_val = mlp.forward(X_val)
    print(classification_report(mlp_val, y_val, label_names))
    _plot(mlp_hist, "MLP", results_dir / "mlp_training.png")

    # --- CNN ---
    print("\nTraining tiny CNN...")
    cnn = TinyCNN(num_classes=5, seed=args.seed)
    cnn_hist = train_mlp(cnn, X_train, y_train, X_val, y_val, epochs=args.epochs, lr=0.1, batch_size=64)
    cnn_val = cnn.forward(X_val)
    print(classification_report(cnn_val, y_val, label_names))
    _plot(cnn_hist, "Tiny CNN", results_dir / "cnn_training.png")

    # --- side-by-side validation accuracy ---
    fig, ax = plt.subplots(figsize=(7, 5))
    epochs = range(1, min(len(mlp_hist["val_acc"]), len(cnn_hist["val_acc"])) + 1)
    ax.plot(epochs, mlp_hist["val_acc"], marker="o", label="MLP")
    ax.plot(epochs, cnn_hist["val_acc"], marker="s", label="Tiny CNN")
    ax.set_title("Validation accuracy: MLP vs Tiny CNN")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.3, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "mlp_vs_cnn.png", dpi=120)
    plt.close(fig)

    summary = (
        f"\nFinal validation accuracy\n"
        f"  MLP      : {mlp_hist['val_acc'][-1]:.4f}\n"
        f"  Tiny CNN : {cnn_hist['val_acc'][-1]:.4f}\n"
        f"Plots saved to {results_dir}/"
    )
    print(summary)


if __name__ == "__main__":
    main()
