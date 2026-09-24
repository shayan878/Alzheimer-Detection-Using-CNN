from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def create_train_test_loaders(
    images,
    labels,
    test_size: float = 0.2,
    random_state: int = 42,
    batch_size: int = 64,
):
    x_train, x_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    train_dataset = list(zip(x_train, y_train))
    test_dataset = list(zip(x_test, y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset
