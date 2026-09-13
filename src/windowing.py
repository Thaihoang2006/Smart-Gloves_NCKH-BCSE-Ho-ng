import pandas as pd
import numpy as np


def create_windows(
    data,
    window_size=50,
    step=25
):
    # Cột đầu tiên là label
    labels = data.iloc[:, 0]

    # Các cột còn lại là features
    features = data.iloc[:, 1:].to_numpy()

    # Tạo mapping label tự động, hỗ trợ nhiều class
    unique_labels = sorted(labels.unique())
    label_to_id = {
        label: idx
        for idx, label in enumerate(unique_labels)
    }

    windows = []
    window_labels = []
    skipped = 0

    for start in range(
        0,
        len(features) - window_size + 1,
        step
    ):
        end = start + window_size

        window = features[start:end]
        label_slice = labels.iloc[start:end]

        # Chỉ giữ window nếu toàn bộ cùng một label
        if label_slice.nunique() != 1:
            skipped += 1
            continue

        windows.append(window)

        label = label_slice.iloc[0]
        window_labels.append(label_to_id[label])

    # Chuyển list thành numpy array
    if windows:
        X = np.stack(windows)
        y = np.asarray(window_labels)
    else:
        X = np.empty(
            (0, window_size, features.shape[1])
        )
        y = np.empty((0,))
    print("Label mapping:", label_to_id)
    return X, y, skipped


if __name__ == "__main__":

    # Đọc dữ liệu
    from pathlib import Path
    import pandas as pd

    ROOT = Path(__file__).resolve().parent.parent
    DATA_PATH = ROOT / "data" / "hand_data.csv"

    data = pd.read_csv(
        DATA_PATH,
        header=None
    )

    # Cấu hình windowing
    window_size = 50
    step = 25

    # Tạo window
    X, y, skipped = create_windows(
        data,
        window_size=window_size,
        step=step
    )

    print("=== WINDOWING RESULT ===")

    print("Kích thước dữ liệu gốc:", data.shape)

    print("Window size:", window_size)
    print("Step:", step)

    print("Số lượng window:", len(X))

    print(
        "Số window bị bỏ vì lẫn nhãn:",
        skipped
    )

    print("Shape X:", X.shape)
    print("Shape y:", y.shape)
    
    # Lưu dữ liệu windowing cho W5
    OUTPUT_DIR = ROOT / "outputs"
    OUTPUT_DIR.mkdir(exist_ok=True)

    np.save(OUTPUT_DIR / "X_windows.npy", X)
    np.save(OUTPUT_DIR / "y_windows.npy", y)

    print("\nĐã lưu:")
    print("X ->", OUTPUT_DIR / "X_windows.npy")
    print("y ->", OUTPUT_DIR / "y_windows.npy")
    # Kiểm tra một số window đầu tiên
    print("\nMột số window đầu:")

    for i in range(min(3, len(X))):
        print(
            f"Window {i + 1}: "
            f"shape = {X[i].shape}, "
            f"label = {y[i]}"
        )