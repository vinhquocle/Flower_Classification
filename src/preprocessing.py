import os
import shutil
import scipy.io
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- CẤU HÌNH ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_and_split_data(source_dir, labels_file, base_dir,
                        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):

    labels_mat = scipy.io.loadmat(labels_file)
    labels = labels_mat['labels'][0]
    filenames = sorted(os.listdir(source_dir))
    filepaths = [os.path.join(source_dir, f) for f in filenames]

    # Chia train - validation - test theo tỉ lệ
    train_files, val_test_files, train_labels, val_test_labels = train_test_split(
        filepaths, labels, train_size=train_ratio, random_state=42, stratify=labels
    )

    val_ratio_in_remaining = val_ratio / (val_ratio + test_ratio)
    val_files, test_files, val_labels, test_labels = train_test_split(
        val_test_files, val_test_labels, train_size=val_ratio_in_remaining,
        random_state=42, stratify=val_test_labels
    )

    print(f"Tong anh: {len(filepaths)}")
    print(f" Train: {len(train_files)}  Validation: {len(val_files)}  Test: {len(test_files)}")

    # Xóa thư mục cũ nếu có
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    # Hàm phụ: copy ảnh vào thư mục phân loại
    def copy_files_to_dir(files, labels, set_name):
        for f, l in zip(files, labels):
            class_dir = os.path.join(base_dir, set_name, str(l - 1))
            os.makedirs(class_dir, exist_ok=True)
            shutil.copy(f, class_dir)

    copy_files_to_dir(train_files, train_labels, "train")
    copy_files_to_dir(val_files, val_labels, "validation")
    copy_files_to_dir(test_files, test_labels, "test")

    return train_files, val_files, test_files

def create_generators(base_dir):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(base_dir, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',   
        color_mode='rgb',      
        shuffle=True
    )
    with open(os.path.join(base_dir, 'cat_to_name.json'), 'w') as f:
        json.dump(train_generator.class_indices, f)

    validation_generator = val_test_datagen.flow_from_directory(
        directory=os.path.join(base_dir, 'validation'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',    
        color_mode='rgb',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        directory=os.path.join(base_dir, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',     
        color_mode='rgb',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator
    

def visualize_augmentation(train_generator, n_samples=9):
    augmented_images, augmented_labels = next(train_generator)
    plt.figure(figsize=(12, 12))
    for i in range(min(n_samples, 9)):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[i])
        class_index = np.argmax(augmented_labels[i]) + 1
        plt.title(f"Class: {class_index}")
        plt.axis("off")
    plt.suptitle("Ví dụ ảnh sau khi tăng cường dữ liệu", fontsize=16)
    plt.show()


#XƯ LÝ DỮ LIỆU ĐỐI VỚI EFFICIENTNET
def create_generators_efficientnet(base_dir):
    # Hàm này dùng preprocess_input và KHÔNG rescale
    def preprocess_rgb_safe(x):
        # Nếu ảnh chỉ có 1 kênh (grayscale) thì chuyển sang RGB
        if x.shape[-1] == 1:
            x = tf.image.grayscale_to_rgb(x)
        return preprocess_input(x)

    train_datagen_eff = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_rgb_safe,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_test_datagen_eff = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_rgb_safe
    )

    train_generator = train_datagen_eff.flow_from_directory(
        directory=os.path.join(base_dir, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True
    )
    with open(os.path.join(base_dir, 'cat_to_name.json'), 'w') as f:
        json.dump(train_generator.class_indices, f)

    validation_generator = val_test_datagen_eff.flow_from_directory(
        directory=os.path.join(base_dir, 'validation'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False
    )

    test_generator = val_test_datagen_eff.flow_from_directory(
        directory=os.path.join(base_dir, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator
