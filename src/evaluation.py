import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import os

def evaluate_model(model_path, test_generator, model_name="Unknown Model"):
    """Đánh giá mô hình trên tập test + trực quan kết quả"""

    # --- Load model ---
    if not os.path.exists(model_path):
        print(f" Không tìm thấy file model: {model_path}")
        return

    model = tf.keras.models.load_model(model_path)
    print(f"\n Đang đánh giá mô hình: {model_name}")
    test_loss, test_acc = model.evaluate(test_generator, verbose=1)
    print(f" Test Accuracy: {test_acc*100:.2f}% | Loss: {test_loss:.4f}")

    # --- Dự đoán ---
    preds_probs = model.predict(test_generator)
    preds = np.argmax(preds_probs, axis=1)
    y_true = test_generator.classes
    class_indices = test_generator.class_indices
   
    # Lấy tên lớp theo đúng thứ tự index
    target_names = [str(k) for k, v in sorted(class_indices.items(), key=lambda item: item[1])]

    # --- Báo cáo ---
    print("\n Classification Report:")
    print(classification_report(y_true, preds, target_names=target_names, zero_division=0))

    # --- Tính các chỉ số tổng quát ---
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, average='macro', zero_division=0)
    rec = recall_score(y_true, preds, average='macro', zero_division=0)
    f1 = f1_score(y_true, preds, average='macro', zero_division=0)
    print(f"\n🔹 Tổng hợp chỉ số ({model_name}):")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    # --- Ma trận nhầm lẫn ---
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, cmap='magma', square=True)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(f'confusion_matrix_{model_name}.png') # Lưu lại
    plt.show()

    # --- Thống kê nhầm lẫn ---
    total_per_class = cm.sum(axis=1)
    correct_per_class = np.diag(cm)
    misclassified_per_class = total_per_class - correct_per_class
    
    # Đảm bảo df_stats dùng target_names đã sắp xếp
    df_stats = pd.DataFrame({
        'Class_ID': target_names, 
        'Total': total_per_class,
        'Correct': correct_per_class,
        'Misclassified': misclassified_per_class
    })
    
    # Tính Accuracy (tránh chia cho 0 nếu lớp nào đó không có ảnh test)
    df_stats['Accuracy_%'] = (df_stats['Correct'] / df_stats['Total'].replace(0, np.nan) * 100)
    df_stats = df_stats.fillna(0) # Thay thế NaN (từ 0/0) bằng 0

    # --- Top 10 chính xác nhất ---
    top10_best = df_stats.nlargest(10, 'Accuracy_%')
    plt.figure(figsize=(10,6))
    sns.barplot(data=top10_best, x='Accuracy_%', y='Class_ID', palette='viridis')
    plt.title(f"Top 10 lớp hoa được dự đoán chính xác nhất - {model_name}")
    plt.xlabel("Độ chính xác (%)"); plt.ylabel("Class ID")
    plt.xlim(0, 100)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig(f'top10_best_{model_name}.png') # Lưu lại
    plt.show()

    # --- Top 10 nhầm nhiều nhất ---
    # Lọc ra các lớp có lỗi > 0
    worst_df = df_stats[df_stats['Misclassified'] > 0]
    top10_worst = worst_df.nlargest(10, 'Misclassified')
    
    plt.figure(figsize=(10,6))
    sns.barplot(data=top10_worst, x='Misclassified', y='Class_ID', palette='rocket')
    plt.title(f"Top 10 lớp hoa bị nhầm nhiều nhất - {model_name}")
    plt.xlabel("Số lượng ảnh bị nhầm"); plt.ylabel("Class ID")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig(f'top10_worst_{model_name}.png') # Lưu lại
    plt.show()

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }