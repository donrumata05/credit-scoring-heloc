import torch.nn as nn
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import torch.optim as optim
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score


class CreditNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def create_xgb():
    return XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=5,
    eval_metric='logloss'
)


def create_logreg():
    return LogisticRegression()


def create_nn(input_dim):
    return CreditNN(input_dim)


def train_nn(model, X_train, X_val, y_train, y_val):
    X_train_t = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train_t = torch.tensor(
        y_train.values,
        dtype=torch.float32
    ).reshape(-1, 1)

    X_val_t = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    y_val_t = torch.tensor(
        y_val.values,
        dtype=torch.float32
    ).reshape(-1, 1)

    train_dataset = TensorDataset(
        X_train_t,
        y_train_t
    )

    val_dataset = TensorDataset(
        X_val_t,
        y_val_t
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False
    )

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    train_losses = []
    val_losses = []
    best_val_f1 = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        train_loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=False
        )

        for batch_X, batch_y in train_loop:
            optimizer.zero_grad()

            prediction = model(batch_X)
            loss = criterion(prediction, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        val_predictions = []
        val_labels = []

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                prediction = model(batch_X)

                loss = criterion(prediction, batch_y)
                val_loss += loss.item()

                val_predictions.extend(
                    prediction.numpy().flatten()
                )
                val_labels.extend(
                    batch_y.numpy().flatten()
                )

        mean_val_loss = val_loss / len(val_loader)
        mean_train_loss = train_loss / len(train_loader)

        val_predictions = np.array(val_predictions)
        val_labels = np.array(val_labels)

        current_f1 = f1_score(
            val_labels,
            (val_predictions > 0.5).astype(int)
        )

        val_losses.append(mean_val_loss)
        train_losses.append(mean_train_loss)

        print(
            f'Epoch: {epoch + 1}/{epochs}, '
            f'train_loss = {mean_train_loss:.4f}, '
            f'val_loss = {mean_val_loss:.4f}, '
            f'val_f1 = {current_f1:.4f}'
        )

        if current_f1 > best_val_f1:
            best_val_f1 = current_f1

            torch.save(
                model.state_dict(),
                'best_model.pth'
            )

            print("Model weights updated")

    model.load_state_dict(
        torch.load(
            'best_model.pth',
            map_location='cpu'
        )
    )

    return model