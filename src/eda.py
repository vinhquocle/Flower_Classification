import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import random

def explore_dataset(source_dir, labels_file):
    """Phân tích EDA cơ bản: phân bố lớp, ảnh mẫu, kích thước ảnh."""

    import scipy.io
    labels_mat = scipy.io.loadmat(labels_file)
    labels = labels_mat['labels'][0]
    filenames = sorted(os.listdir(source_dir))
    filepaths = [os.path.join(source_dir, f) for f in filenames]

    # --- Phân bố lớp ---
    label_counts = pd.Series(labels).value_counts().sort_index()
    plt.figure(figsize=(12, 4))
    sns.barplot(x=label_counts.index, y=label_counts.values, color='skyblue')
    plt.title("Phân bố số lượng ảnh trong từng lớp hoa")
    plt.xlabel("Class ID")
    plt.ylabel("Số ảnh")
    plt.show()

    print(f"Tổng số lớp: {len(label_counts)}")
    print(f"Trung bình số ảnh mỗi lớp: {label_counts.mean():.2f}")
    print(f"Lớp có ít nhất ảnh: {label_counts.idxmin()} ({label_counts.min()} ảnh)")
    print(f"Lớp có nhiều nhất ảnh: {label_counts.idxmax()} ({label_counts.max()} ảnh)")

    # --- Hiển thị ảnh ngẫu nhiên ---
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        idx = random.randint(0, len(filepaths) - 1)
        img = mpimg.imread(filepaths[idx])
        ax.imshow(img)
        ax.set_title(f"Class {labels[idx]}")
        ax.axis("off")
    plt.suptitle("Một vài ảnh mẫu ngẫu nhiên trong dataset", fontsize=14)
    plt.show()

    # --- Thống kê kích thước ảnh ---
    sizes = []
    for path in np.random.choice(filepaths, size=min(200, len(filepaths)), replace=False):
        img = Image.open(path)
        sizes.append(img.size)
    sizes_df = pd.DataFrame(sizes, columns=['width', 'height'])
    print(sizes_df.describe())

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x='width', y='height', data=sizes_df)
    plt.title("Phân bố kích thước ảnh (mẫu 200 ảnh)")
    plt.show()
