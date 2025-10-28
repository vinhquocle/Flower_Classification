import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report # Import đã có
import seaborn as sns
import numpy as np
import pandas as pd

def train_cnn_advanced(train_generator, val_generator, test_generator, num_classes=102):
    """
    Hàm này xây dựng, huấn luyện và đánh giá mô hình CNN.
    """
    
    # --- Xây dựng mô hình CNN cải tiến ---
    model = models.Sequential([
        layers.Conv2D(32, (5,5), activation='relu', input_shape=(224,224,3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(256, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # --- Callback ---
    checkpoint = ModelCheckpoint("best_cnn.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)

    # --- Huấn luyện ---
    # Chạy tối đa 50 epochs, nhưng EarlyStopping sẽ dừng sớm nếu model không cải thiện
    history = model.fit(
        train_generator,
        epochs=50, # Comment đã sửa cho khớp (trước là 100)
        validation_data=val_generator,
        callbacks=[checkpoint, earlystop]
    )

    # --- Vẽ biểu đồ Accuracy và Loss ---
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc)) # Số epochs thực tế đã chạy

    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, acc, label='Train Acc')
    plt.plot(epochs_range, val_acc, label='Val Acc')
    plt.legend(); plt.title('CNN Accuracy'); plt.xlabel('Epochs'); plt.ylabel('Accuracy')

    plt.subplot(1,2,2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.legend(); plt.title('CNN Loss'); plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.tight_layout(); plt.show()


    
    # --- Evaluate trên test set ---
    model.load_weights("best_cnn.h5") # Tải lại model tốt nhất đã lưu
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"\nTest Accuracy (CNN): {test_acc*100:.2f}% | Loss: {test_loss:.4f}")

    # --- Ma trận nhầm lẫn & Báo cáo chi tiết ---
    # Reset generator trước khi predict để đảm bảo thứ tự
    test_generator.reset() 
    
    predictions = model.predict(test_generator, steps=len(test_generator))
    preds = np.argmax(predictions, axis=1)
    y_true = test_generator.classes 
    
    # Lấy tên các lớp, sắp xếp theo đúng thứ tự index
    sorted_indices = sorted(test_generator.class_indices.items(), key=lambda item: item[1])
    class_labels = [name for name, index in sorted_indices]
    #Vẽ Ma trận nhầm lẫn
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, cmap='magma', square=True)
    plt.title("Confusion Matrix - CNN")
    plt.xlabel("Predicted Label"); plt.ylabel("True Label")
    plt.show()
    
    #  In Báo cáo phân loại (Classification Report)
    print("\n--- Báo cáo phân loại chi tiết (CNN) ---")
    print(classification_report(y_true, preds, target_names=class_labels))

    # Top 10 lớp bị nhầm nhiều nhất
    total_per_class = cm.sum(axis=1)
    correct_per_class = np.diag(cm)
    misclassified_per_class = total_per_class - correct_per_class
    
    df_stats = pd.DataFrame({
        'Class_ID': list(range(len(total_per_class))),
        'Class_Name': class_labels, 
        'Total': total_per_class,
        'Correct': correct_per_class,
        'Misclassified': misclassified_per_class
    })
    
    top10 = df_stats.nlargest(10, 'Misclassified')
    print("\nTop 10 lớp bị nhầm lẫn nhiều nhất (CNN):")
    print(top10[['Class_Name', 'Misclassified', 'Total']]) 

    return model, history, df_stats