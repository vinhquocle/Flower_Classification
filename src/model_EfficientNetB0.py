import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0 as EfficientNetB0
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import os 


def train_efficientnet(train_generator, val_generator, test_generator, num_classes=102):
    """Huấn luyện mô hình EfficientNetB0 (Transfer Learning + Fine-tuning)"""

    # 1. Tải base model
    base_model = EfficientNetB0(input_shape=(224,224,3), 
                                include_top=False, 
                                weights='imagenet')
    
    # 2. Đóng băng base model
    base_model.trainable = False

    # 3. Xây dựng phần "đầu" (head) của model
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x) # Giữ nguyên cấu trúc head giống MobileNetV2
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)

    # 4. Compile cho Transfer Learning
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    model.summary()

    # 5. Callbacks
    checkpoint_path = "best_efficientnet.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, 
                                 monitor='val_accuracy', 
                                 save_best_only=True, 
                                 verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', 
                              patience=5, 
                              restore_best_weights=True, 
                              verbose=1)

    # 6. Train lần đầu (Transfer Learning)
    print("\n--- Giai đoạn 1: Huấn luyện Transfer Learning ---")
    history = model.fit(train_generator, 
                        epochs=50, 
                        validation_data=val_generator, 
                        callbacks=[checkpoint, earlystop])

    # 7. Chuẩn bị Fine-tuning

    # Mở băng base model
    base_model.trainable = True
    
    # Đóng băng các lớp đầu tiên
    # EfficientNetB0 có 239 lớp. Đóng băng 200 lớp đầu tiên
    # và chỉ fine-tune các lớp cuối.
    for layer in base_model.layers[:200]:
        layer.trainable = False
        
    # Compile lại với learning rate cực nhỏ
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Xác định epoch bắt đầu (ví dụ: 50)
    initial_epoch_fine = len(history.history['loss']) 
    
    # Tính tổng số epochs (ví dụ: 100)
    total_epochs_fine = initial_epoch_fine + 50 # Chạy thêm 50 epochs
    
    history_fine = model.fit(train_generator, 
                             epochs=total_epochs_fine, 
                             initial_epoch=initial_epoch_fine, 
                             validation_data=val_generator, 
                             callbacks=[checkpoint, earlystop])

    # --- 8. Vẽ biểu đồ ---
    # Nối 2 giai đoạn history
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, acc, label='Train Acc')
    plt.plot(epochs_range, val_acc, label='Val Acc')
    plt.axvline(x=initial_epoch_fine - 1, color='gray', linestyle='--', label='Fine-tuning start')
    plt.legend(); plt.title('EfficientNetB0 Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy')

    plt.subplot(1,2,2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.axvline(x=initial_epoch_fine - 1, color='gray', linestyle='--')
    plt.legend(); plt.title('EfficientNetB0 Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig("efficientnet_training_plot.png") # Lưu biểu đồ
    plt.show()

    # --- 9. Đánh giá ---
    print("\n--- Đánh giá trên tập Test (EfficientNetB0) ---")
    model.load_weights(checkpoint_path) # Tải model tốt nhất
    test_loss, test_acc = model.evaluate(test_generator)
    print(f" Test Accuracy (EfficientNetB0): {test_acc*100:.2f}% | Loss: {test_loss:.4f}")

    # --- 10. Ma trận nhầm lẫn & Báo cáo ---
    test_generator.reset() # Reset generator
    preds = np.argmax(model.predict(test_generator, steps=len(test_generator)), axis=1)
    y_true = test_generator.classes
    
    # Lấy tên lớp
    class_labels = list(test_generator.class_indices.keys())
    
    # Báo cáo
    print("\n--- Báo cáo phân loại (EfficientNetB0) ---")
    print(classification_report(y_true, preds, target_names=class_labels))

    # Ma trận nhầm lẫn
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, cmap='cividis', square=True) # Đổi màu cho khác
    plt.title("Confusion Matrix - EfficientNetB0")
    plt.xlabel("Predicted Label"); plt.ylabel("True Label")
    plt.savefig("efficientnet_confusion_matrix.png") # Lưu biểu đồ
    plt.show()

    # --- 11. Thống kê Top 10 nhầm lẫn ---
    total_per_class = cm.sum(axis=1)
    correct_per_class = np.diag(cm)
    misclassified_per_class = total_per_class - correct_per_class
    
    df_stats = pd.DataFrame({
        'Class_ID': [int(cls) for cls in class_labels], 
        'Class_Name': class_labels,
        'Total': total_per_class,
        'Correct': correct_per_class,
        'Misclassified': misclassified_per_class
    })
    
    top10 = df_stats.sort_values(by='Misclassified', ascending=False).head(10)
    print("\n Top 10 lớp bị nhầm lẫn nhiều nhất (EfficientNetB0):")
    print(top10)

    return model, history # Trả về model và history