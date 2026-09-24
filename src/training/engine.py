import torch
from torchmetrics import Accuracy
from tqdm import tqdm

from .metrics import AverageMeter


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    loss_meter = AverageMeter()
    accuracy = Accuracy(task="multiclass", num_classes=2).to(device)

    with tqdm(train_loader, unit="batch") as progress:
        for data, target in progress:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), data.size(0))
            accuracy.update(output, target.int())
            progress.set_postfix(
                loss=loss_meter.avg,
                accuracy=100.0 * accuracy.compute().item(),
            )

    return model, loss_meter.avg, accuracy.compute().item()


def evaluate(model, device, test_loader, criterion):
    model.eval()
    loss_meter = AverageMeter()
    accuracy = Accuracy(task="multiclass", num_classes=2).to(device)

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss_meter.update(loss.item(), inputs.size(0))
            accuracy.update(outputs, targets.int())

    return loss_meter.avg, accuracy.compute().item()


def train_and_evaluate_model(
    model,
    optimizer,
    epochs,
    device,
    train_loader,
    test_loader,
    loss_fn,
):
    history = {
        "train_loss": [],
        "test_loss": [],
        "train_accuracy": [],
        "test_accuracy": [],
    }

    for epoch in range(1, epochs + 1):
        model, train_loss, train_acc = train(
            model, device, train_loader, optimizer, loss_fn
        )
        test_loss, test_acc = evaluate(
            model, device, test_loader, loss_fn
        )

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_accuracy"].append(train_acc)
        history["test_accuracy"].append(test_acc)

        print(
            f"Epoch {epoch}, Test: "
            f"Loss={test_loss:.4f}, Accuracy={test_acc:.4f}"
        )

    return history
