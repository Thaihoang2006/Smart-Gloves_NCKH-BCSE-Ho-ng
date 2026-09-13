import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path)
        self.y = np.load(y_path)

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
            return self.X[index], self.y[index]
if __name__ == "__main__":
    dataset = WindowDataset(
        "outputs/X_windows.npy",
        "outputs/y_windows.npy"
    )

    print("=== WINDOW DATASET RESULT ===")
    print("Số lượng window:", len(dataset))
    print("Shape X:", dataset.X.shape)
    print("Shape y:", dataset.y.shape)

    x, y = dataset[0]

    print("Shape window đầu tiên:", x.shape)
    print("Label window đầu tiên:", y.item())
    