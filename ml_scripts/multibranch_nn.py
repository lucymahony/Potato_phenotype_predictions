import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import os

class PreprocessedMultiOmicDataset(Dataset):
    def __init__(self, lc, gc, trans, targets):
        self.lc = lc
        self.gc = gc
        self.trans = trans
        self.targets = targets

    def __len__(self):
        return self.lc.shape[0]

    def __getitem__(self, idx):
        return (self.lc[idx], self.gc[idx], self.trans[idx]), self.targets[idx]


class MultiBranchNet(nn.Module):
    def __init__(self, input_sizes, hidden_size=128, output_size=1):
        super().__init__()
        lc_size, gc_size, trans_size = input_sizes

        self.lc_branch = nn.Sequential(
            nn.Linear(lc_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
        )
        self.gc_branch = nn.Sequential(
            nn.Linear(gc_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
        )
        self.trans_branch = nn.Sequential(
            nn.Linear(trans_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_size)
        )

    def forward(self, x_lc, x_gc, x_trans):
        out_lc = self.lc_branch(x_lc)
        out_gc = self.gc_branch(x_gc)
        out_trans = self.trans_branch(x_trans)
        merged = torch.cat([out_lc, out_gc, out_trans], dim=1)
        return self.head(merged)


def train_model(model, dataset, device, output_file_path, epochs=100, batch_size=6, lr=1e-3, patience=10):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Starting training with lr={lr}...\n")
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for (x_lc, x_gc, x_trans), y in loader:
            x_lc, x_gc, x_trans, y = x_lc.to(device), x_gc.to(device), x_trans.to(device), y.to(device)
            preds = model(x_lc, x_gc, x_trans)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(loader)
        print(f"Epoch {epoch + 1:02d} | Avg Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), output_file_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break


def test_model(model, test_dataset, device, output_file_path, batch_size=6):
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.load_state_dict(torch.load(output_file_path))
    model.eval()
    model = model.to(device)
    criterion = nn.MSELoss()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for (x_lc, x_gc, x_trans), y in loader:
            x_lc, x_gc, x_trans, y = x_lc.to(device), x_gc.to(device), x_trans.to(device), y.to(device)
            preds = model(x_lc, x_gc, x_trans)
            loss = criterion(preds, y)
            total_loss += loss.item()
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    avg_loss = total_loss / len(loader)
    print(f"Test Loss: {avg_loss:.4f}")
    return torch.cat(all_preds), torch.cat(all_targets)


def plot_results(preds, targets, output_file_path):
    preds_np = preds.flatten()
    targets_np = targets.flatten()
    r2 = r2_score(targets_np, preds_np)
    print(f"R² score: {r2:.4f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(targets_np, preds_np, alpha=0.7)
    plt.plot([targets_np.min(), targets_np.max()], [targets_np.min(), targets_np.max()], 'r--')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Prediction vs True | R² = {r2:.2f}")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    plt.savefig(output_file_path)
    return r2


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load('../intermediate_data/preprocessed_dataset.pt')
    target_keys = [['colour'], ['tubershape'], ['decol5min'], ['DSC_Onset']]
    results = []

    for key in target_keys:
        raw_targets = torch.tensor([[t[k] for k in key] for t in data['phenotypes']], dtype=torch.float32)
        scaler = StandardScaler()
        target_scaled = scaler.fit_transform(raw_targets.numpy())
        target_tensor = torch.tensor(target_scaled, dtype=torch.float32)

        dataset = PreprocessedMultiOmicDataset(data['lc'], data['gc'], data['trans'], target_tensor)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        for lr in [1e-3, 1e-4, 3e-4, 3e-5, 3e-2]:
            print(f"\n--- Training {key[0]} with learning rate: {lr} ---")
            model = MultiBranchNet(
                input_sizes=(dataset.lc.shape[1], dataset.gc.shape[1], dataset.trans.shape[1]),
                output_size=1
            )
            model_file = f"../intermediate_data/model_{key[0]}_lr{lr}.pth"
            train_model(model, train_dataset, device, model_file, epochs=100, batch_size=6, lr=lr, patience=10)

            preds, targets = test_model(model, test_dataset, device, model_file, batch_size=6)
            preds = scaler.inverse_transform(preds.numpy())
            targets = scaler.inverse_transform(targets.numpy())
            r2 = plot_results(torch.tensor(preds), torch.tensor(targets), f"../plots/plot_{key[0]}_lr{lr}.png")
            results.append({"target": key[0], "lr": lr, "r2": r2})

    df_results = pd.DataFrame(results)
    df_results.to_csv("../intermediate_data/model_r2_scores.csv", index=False)
