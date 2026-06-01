"""
Eğitim sonrası grafikler — yalnızca presentation/artifacts/training_results.json okur.
Önce: python presentation/train_bccd_classifier.py

Çalıştırma: python presentation/generate_training_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
RESULTS_PATH = ARTIFACTS / "training_results.json"
OUTPUT_DIR = ROOT / "figures"

HEALTH_COLORS = {
    "primary": "#1a365d",
    "wbc": "#2e7d32",
    "rbc": "#c62828",
    "plt": "#1565c0",
    "bg": "#f7f9fc",
}


def load_results() -> dict:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Eğitim çıktısı bulunamadı: {RESULTS_PATH}\n"
            "Önce çalıştırın: python presentation/train_bccd_classifier.py"
        )
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Segoe UI",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "figure.facecolor": HEALTH_COLORS["bg"],
            "axes.facecolor": "white",
        }
    )


def plot_training_curves(results: dict) -> plt.Figure:
    history = results["history"]
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    class_names = ", ".join(results["class_names"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    axes[0].plot(epochs, train_loss, "o-", color=HEALTH_COLORS["wbc"], lw=2, ms=4, label="Eğitim")
    axes[0].plot(epochs, val_loss, "s--", color=HEALTH_COLORS["rbc"], lw=2, ms=4, label="Doğrulama")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Kayıp (Cross-Entropy)")
    axes[0].set_title("Gerçek Eğitim / Doğrulama Kaybı")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_acc, "o-", color=HEALTH_COLORS["wbc"], lw=2, ms=4, label="Eğitim")
    axes[1].plot(epochs, val_acc, "s--", color=HEALTH_COLORS["rbc"], lw=2, ms=4, label="Doğrulama")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Doğruluk (%)")
    axes[1].set_title("Gerçek Eğitim / Doğrulama Doğruluğu")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 100)

    fig.suptitle(
        f"BCCD Hücre Sınıflandırma — Gerçek Eğitim Logları ({class_names})",
        fontsize=14,
        fontweight="bold",
        color=HEALTH_COLORS["primary"],
        y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_confusion_matrix(results: dict) -> plt.Figure:
    class_names = results["class_names"]
    cm = np.array(results["confusion_matrix"], dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums > 0) * 100
    test_acc = results["test_accuracy_percent"]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Satır içi oran (%)"},
        linewidths=0.8,
        linecolor="white",
        ax=ax,
        vmin=0,
        vmax=100,
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j + 0.5,
                i + 0.72,
                f"n={int(cm[i, j])}",
                ha="center",
                va="center",
                fontsize=8,
                color="#444",
            )

    ax.set_xlabel("Tahmin Edilen Sınıf", fontweight="bold")
    ax.set_ylabel("Gerçek Sınıf", fontweight="bold")
    ax.set_title(
        f"Confusion Matrix — BCCD Test Seti | Doğruluk: {test_acc:.2f}%",
        fontweight="bold",
        color=HEALTH_COLORS["primary"],
        pad=12,
    )
    fig.tight_layout()
    return fig


def plot_combined_summary(results: dict) -> plt.Figure:
    history = results["history"]
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    class_names = results["class_names"]
    cm = np.array(results["confusion_matrix"], dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums > 0) * 100
    test_acc = results["test_accuracy_percent"]
    best_val = results["best_val_acc_percent"]
    n_epochs = results["epochs"]

    fig = plt.figure(figsize=(14, 8), dpi=150)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.1], hspace=0.35, wspace=0.28)

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_acc = fig.add_subplot(gs[0, 1])
    ax_cm = fig.add_subplot(gs[1, :])

    ax_loss.plot(epochs, train_loss, "o-", color=HEALTH_COLORS["wbc"], lw=2, ms=3, label="Eğitim")
    ax_loss.plot(epochs, val_loss, "s--", color=HEALTH_COLORS["rbc"], lw=2, ms=3, label="Doğrulama")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Kayıp Eğrisi (gerçek)")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, train_acc, "o-", color=HEALTH_COLORS["wbc"], lw=2, ms=3, label="Eğitim")
    ax_acc.plot(epochs, val_acc, "s--", color=HEALTH_COLORS["rbc"], lw=2, ms=3, label="Doğrulama")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Doğruluk Eğrisi (gerçek)")
    ax_acc.legend(loc="lower right")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_ylim(0, 100)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "%"},
        ax=ax_cm,
        linewidths=0.6,
        linecolor="white",
    )
    ax_cm.set_xlabel("Tahmin")
    ax_cm.set_ylabel("Gerçek")
    ax_cm.set_title("Confusion Matrix — Test (gerçek sayım)")

    fig.suptitle(
        f"BCCD Kan Hücresi CNN — Gerçek Eğitim Özeti | "
        f"Test: {test_acc:.1f}% | En iyi Val: {best_val:.1f}% | Epoch: {n_epochs}",
        fontsize=15,
        fontweight="bold",
        color=HEALTH_COLORS["primary"],
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        results.get("note", ""),
        ha="center",
        fontsize=9,
        color="#6b7c93",
        style="italic",
    )
    return fig


def main() -> None:
    setup_style()
    results = load_results()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig1 = plot_training_curves(results)
    path1 = OUTPUT_DIR / "training_curves.png"
    fig1.savefig(path1, bbox_inches="tight", facecolor=fig1.get_facecolor())
    plt.close(fig1)

    fig2 = plot_confusion_matrix(results)
    path2 = OUTPUT_DIR / "confusion_matrix.png"
    fig2.savefig(path2, bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close(fig2)

    fig3 = plot_combined_summary(results)
    path3 = OUTPUT_DIR / "ml_training_summary.png"
    fig3.savefig(path3, bbox_inches="tight", facecolor=fig3.get_facecolor())
    plt.close(fig3)

    print("Gerçek eğitim grafikleri kaydedildi:")
    print(f"  Test doğruluğu: {results['test_accuracy_percent']:.2f}%")
    print(f"  - {path1}")
    print(f"  - {path2}")
    print(f"  - {path3}")


if __name__ == "__main__":
    main()
