import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def train_mobilenet(train_generator, val_generator, test_generator, num_classes=102):
    """Huấn luyện mô hình MobileNetV2 (Transfer Learning + Fine-tuning)"""

    base_model = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
    base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    checkpoint = ModelCheckpoint("best_mobilenet.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    # Train lần đầu (50 epoch)
    history = model.fit(train_generator, 
    epochs=50, 
    validation_data=val_generator, 
    callbacks=[checkpoint, earlystop])

    # Fine-tuning 
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    history_fine = model.fit(train_generator, epochs=100, initial_epoch=50, validation_data=val_generator, callbacks=[checkpoint, earlystop])

    # --- Biểu đồ ---
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, acc, label='Train Acc')
    plt.plot(epochs_range, val_acc, label='Val Acc')
    plt.axvline(x=49, color='gray', linestyle='--', label='Fine-tuning start')
    plt.legend(); plt.title('MobileNetV2 Accuracy')
    plt.subplot(1,2,2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.axvline(x=49, color='gray', linestyle='--')
    plt.legend(); plt.title('MobileNetV2 Loss')
    plt.tight_layout(); plt.show()

    # --- Đánh giá ---
    model.load_weights("best_mobilenet.h5")
    test_loss, test_acc = model.evaluate(test_generator)
    print(f" Test Accuracy: {test_acc*100:.2f}% | Loss: {test_loss:.4f}")

    # --- Confusion matrix ---
    preds = np.argmax(model.predict(test_generator), axis=1)
    y_true = test_generator.classes
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, cmap='viridis', square=True)
    plt.title("Confusion Matrix - MobileNetV2")
    plt.xlabel("Predicted Label"); plt.ylabel("True Label")
    plt.show()

    # --- Thống kê Top 10 nhầm lẫn ---
    total_per_class = cm.sum(axis=1)
    correct_per_class = np.diag(cm)
    misclassified_per_class = total_per_class - correct_per_class
    df_stats = pd.DataFrame({
        'Class_ID': list(range(len(total_per_class))),
        'Total': total_per_class,
        'Correct': correct_per_class,
        'Misclassified': misclassified_per_class
    })
    top10 = df_stats.nlargest(10, 'Misclassified')
    print("\n Top 10 lớp bị nhầm lẫn nhiều nhất:")
    print(top10)
