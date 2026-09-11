import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class SmartGloveDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        window_size: int = 50,
    ):
        self.data_dir = data_dir
        self.split = split
        self.window_size = window_size

        # Đọc dữ liệu từ file CSV
        file_path = os.path.join(data_dir, "hand_data.csv")
        data = pd.read_csv(file_path, header=None)

        # Cột đầu tiên là label
        self.labels = data.iloc[:, 0]

        # Các cột còn lại là features
        self.features = data.iloc[:, 1:]

        # Tự xác định số lượng feature
        self.input_dim = self.features.shape[1]

        # Số lượng label khác nhau
        self.output_dim = self.labels.nunique()

        # Chuyển features sang Tensor
        self.samples = torch.tensor(
            self.features.to_numpy(),
            dtype=torch.float32
        )

        # Chuyển label sang Tensor
        self.labels_tensor = torch.tensor(
            pd.factorize(self.labels)[0],
            dtype=torch.long
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels_tensor[idx]

        return sample, label


if __name__ == "__main__":
    train_set = SmartGloveDataset(
        data_dir="./data",
        split="train"
    )

    print("Số lượng mẫu:", len(train_set))
    print("Số feature:", train_set.input_dim)
    print("Số loại label:", train_set.output_dim)

    sample, label = train_set[0]

    print("Shape sample đầu tiên:", sample.shape)
    print("Label đầu tiên:", label.item())