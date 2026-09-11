import pandas as pd
import matplotlib.pyplot as plt


# Đọc file dữ liệu
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "hand_data.csv"

data = pd.read_csv(DATA_PATH, header=None)

print("Kích thước dữ liệu:", data.shape)
print(data.head())

print("\nTổng số giá trị NaN:", data.isna().sum().sum())

numeric_data = data.select_dtypes(include="number")

print("\nGiá trị nhỏ nhất mỗi cột:")
print(numeric_data.min())

print("\nGiá trị lớn nhất mỗi cột:")
print(numeric_data.max())

print("\nGiá trị trung bình mỗi cột:")
print(numeric_data.mean())

plt.figure(figsize=(20, 15))

plt.plot(data.iloc[:, 1], label="Sensor 1")
plt.plot(data.iloc[:, 2], label="Sensor 2")
plt.plot(data.iloc[:, 3], label="Sensor 3")

plt.title("Visualize Smart Glove Data")
plt.xlabel("Sample")
plt.ylabel("Value")
plt.legend()
plt.grid()

# Lưu biểu đồ
plt.savefig("sensor_plot.png")
print("Đã lưu biểu đồ: sensor_plot.png")