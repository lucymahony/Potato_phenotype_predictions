import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

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
        
        # Unpack input sizes for each modality
        lc_size, gc_size, trans_size = input_sizes

        # Define each branch
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

        # Merge and predict
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


def train_model(model, dataset, device, output_file_path, epochs=10, batch_size=6, lr=1e-3):

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Starting training...\n")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for (x_lc, x_gc, x_trans), y in loader:
            x_lc, x_gc, x_trans, y = x_lc.to(device), x_gc.to(device), x_trans.to(device), y.to(device)

            preds = model(x_lc, x_gc, x_trans)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(loader)
        print(f"Epoch {epoch + 1:02d} | Avg Loss: {epoch_loss:.4f}")
        torch.save(model.state_dict(), output_file_path)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load('../intermediate_data/preprocessed_dataset.pt')

    print(data)
    
    target_keys = [['colour'], ['tubershape'], ['decol5min'], ['DSC_Onset']]
    for key in target_keys:
        target_tensor = torch.tensor([[t[k] for k in key] for t in data['phenotypes']], dtype=torch.float32)
        dataset = PreprocessedMultiOmicDataset(data['lc'], data['gc'], data['trans'], target_tensor)

        model = MultiBranchNet(
            input_sizes=(dataset.lc.shape[1], dataset.gc.shape[1], dataset.trans.shape[1]),
            output_size=1  
        )
        output_file_path = f"../intermediate_data/model_{key[0]}.pth"
        train_model(model, dataset, device, output_file_path, epochs=100, batch_size=6, lr=1e-3)