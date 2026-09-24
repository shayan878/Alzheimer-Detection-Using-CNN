from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import Config
from src.data.dataset import ADNMIDataset
from src.data.loaders import create_train_test_loaders
from src.evaluation.metrics import calculate_metrics, get_roc_data
from src.models.cnn_models import build_model
from src.training.engine import train_and_evaluate_model
from src.utils.seed import set_seed
from src.visualization.plots import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_history,
    plot_roc,
    plot_train_test_distribution,
    show_augmented_images,
    show_original_images,
)


def run_experiment(config: Config):
    set_seed(config.random_state)
    print(f"Using device: {config.device}")

    dataset = ADNMIDataset(
        config.data_dir,
        image_size=config.image_size,
        augmentation_repeats=config.augmentation_repeats,
        class_names=config.class_names,
    )
    print(f"Total samples after augmentation: {len(dataset)}")

    show_original_images(dataset)
    show_augmented_images(dataset)
    plot_class_distribution(
        dataset.labels_orig,
        config.class_names,
        "Class Distribution Before Augmentation",
    )
    plot_class_distribution(
        dataset.labels,
        config.class_names,
        "Class Distribution After Augmentation",
    )

    train_loader, test_loader, train_dataset, test_dataset = create_train_test_loaders(
        dataset.img_data,
        dataset.labels,
        test_size=config.test_size,
        random_state=config.random_state,
        batch_size=config.batch_size,
    )
    plot_train_test_distribution(
        train_dataset,
        test_dataset,
        config.class_names,
    )

    loss_fn = nn.CrossEntropyLoss()
    results = {}

    for model_name in ("proposed", "testing_model1", "testing_model2"):
        print(f"\n===== {model_name} =====")
        model = build_model(model_name, config.device)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        history = train_and_evaluate_model(
            model,
            optimizer,
            config.epochs,
            config.device,
            train_loader,
            test_loader,
            loss_fn,
        )

        plot_history(history, model_name)

        fpr, tpr, roc_auc = get_roc_data(model, test_loader, config.device)
        plot_roc(fpr, tpr, roc_auc)

        evaluation = calculate_metrics(
            model,
            test_loader,
            config.device,
        )

        print("Classification Report:")
        print(evaluation.classification_report)
        print(f"Accuracy: {evaluation.accuracy:.4f}")
        print(f"Precision: {evaluation.precision:.4f}")
        print(f"AUC Score: {evaluation.auc_score:.4f}")
        print(f"Jaccard Score: {evaluation.jaccard:.4f}")
        print(f"Recall: {evaluation.recall:.4f}")
        print(f"Dice Similarity Coefficient: {evaluation.dice:.4f}")

        plot_confusion_matrix(
            evaluation.confusion_matrix,
            class_names=config.class_names,
        )

        results[model_name] = {
            "model": model,
            "history": history,
            "evaluation": evaluation,
        }

    return results


if __name__ == "__main__":
    # Update this path for your environment.
    DATA_DIR = Path("/content/drive/My Drive/ADNI_Dataset22")
    config = Config(data_dir=DATA_DIR)
    run_experiment(config)
