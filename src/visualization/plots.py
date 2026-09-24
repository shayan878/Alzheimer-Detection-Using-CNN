import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def show_original_images(dataset, n=5):
    n = min(n, len(dataset.orig_img))
    indices = random.sample(range(len(dataset.orig_img)), n)

    plt.figure(figsize=(3 * n, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, n, i + 1)
        plt.imshow(np.array(dataset.orig_img[idx]).transpose(1, 2, 0))
        plt.title(f"Label: {dataset.labels_orig[idx]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_augmented_images(dataset, n=5):
    n = min(n, len(dataset.aug_img))
    indices = random.sample(range(len(dataset.aug_img)), n)

    plt.figure(figsize=(3 * n, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, n, i + 1)
        plt.imshow(np.array(dataset.aug_img[idx]).transpose(1, 2, 0))
        plt.title("Augmented")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_class_distribution(labels, class_names, title):
    counts = [labels.count(i) for i in range(len(class_names))]

    plt.figure(figsize=(6, 4))
    plt.bar(class_names, counts)
    plt.xlabel("Classes")
    plt.ylabel("Number of Samples")
    plt.title(title)
    plt.show()

    return counts


def plot_train_test_distribution(train_dataset, test_dataset, class_names):
    train_labels = [label for _, label in train_dataset]
    test_labels = [label for _, label in test_dataset]

    train_counts = [train_labels.count(i) for i in range(len(class_names))]
    test_counts = [test_labels.count(i) for i in range(len(class_names))]

    print(f"\t\t{class_names[0]}\t\t{class_names[1]}")
    print(f"training:\t{train_counts[0]}\t\t{train_counts[1]}")
    print(f"testing:\t{test_counts[0]}\t\t{test_counts[1]}")

    return train_counts, test_counts


def plot_history(history, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_accuracy"], label="Training Accuracy")
    plt.plot(history["test_accuracy"], label="Testing Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Training and Testing Accuracy for {model_name}")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["test_loss"], label="Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Testing Loss for {model_name}")
    plt.legend()
    plt.show()


def plot_roc(fpr, tpr, roc_auc, title="Receiver Operating Characteristic"):
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(conf_mat, class_names=("AD", "MCI")):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt="g",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.show()
