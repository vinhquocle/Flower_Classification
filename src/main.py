import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.eda import explore_dataset
from src.preprocessing import load_and_split_data, create_generators, visualize_augmentation, create_generators_efficientnet
from src.model_cnn import train_cnn_advanced
from src.model_mobileNetV2 import train_mobilenet
from src.model_EfficientNetB0 import train_efficientnet
from src.evaluation import evaluate_model
import tensorflow as tf
import random
import glob
import json
import numpy as np
from tensorflow.keras.preprocessing import image


# CẤU HÌNH ĐƯỜNG DẪN 
SOURCE_DIR = r"C:\Users\vinhq\Downloads\Flower_classification\data\102flowers\jpg"
LABELS_FILE = r"C:\Users\vinhq\Downloads\Flower_classification\data\imagelabels.mat"
BASE_DIR = "flower_data_split"
#EDA
print(" (EDA) ")
explore_dataset(SOURCE_DIR, LABELS_FILE) 
#  CHUẨN BỊ DỮ LIỆU 
if not os.path.exists(BASE_DIR):
    load_and_split_data(SOURCE_DIR, LABELS_FILE, BASE_DIR)
train_gen, val_gen, test_gen = create_generators(BASE_DIR)
visualize_augmentation(train_gen)
train_gen_eff, val_gen_eff, test_gen_eff = create_generators_efficientnet(BASE_DIR)

#  HUẤN LUYỆN MÔ HÌNH
print("\n TRAINING MOBILENETV2 ")
train_mobilenet(train_gen, val_gen, test_gen)
print("\n TRAINING EficientNetB0")
train_efficientnet(train_gen_eff, val_gen_eff, test_gen_eff)
print("\n TRAINING CNN")
train_cnn_advanced(train_gen, val_gen, test_gen)

#ĐÁNH GIÁ MÔ HÌNH 
results = []
results.append(evaluate_model("best_efficientnet.h5", test_gen_eff, model_name="EfficientNetB0"))
results.append(evaluate_model("best_mobilenet.h5", test_gen, model_name="MobileNetV2"))
results.append(evaluate_model("best_cnn.h5", test_gen, model_name="CNN cải tiến"))


#  DỰ ĐOÁN ẢNH NGẪU NHIÊN (SỬ DỤNG MOBILENETV2) ---
print("\n--- 5. Chạy dự đoán trên 1 ảnh ngẫu nhiên (MobileNetV2) ---")

# --- Cấu hình cho dự đoán ---
MODEL_PATH_PREDICT = "best_mobilenet.h5" # Dùng model MobileNet
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "cat_to_name.json")
TEST_DIR_PATH = os.path.join(BASE_DIR, "test")
IMG_SIZE = (224, 224)

def load_and_preprocess_image_mobilenet(img_path):
    try:
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = img_batch / 255.0 # Tiền xử lý của MobileNet
        return img_preprocessed
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {img_path}: {e}")
        return None

try:
    # 1. Tải model
    model_pred = tf.keras.models.load_model(MODEL_PATH_PREDICT)
    
    # 2. Tải class labels
    with open(CLASS_INDICES_PATH) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    # 3. Chọn ảnh ngẫu nhiên
    all_test_images = glob.glob(os.path.join(TEST_DIR_PATH, "*", "*.jpg"))
    if not all_test_images:
        print(f"Lỗi: Không tìm thấy ảnh nào trong {TEST_DIR_PATH}")
    else:
        image_path_to_test = random.choice(all_test_images)
        
        # 4. Tiền xử lý ảnh
        processed_image = load_and_preprocess_image_mobilenet(image_path_to_test)
        
        if processed_image is not None:
            # 5. Dự đoán
            predictions = model_pred.predict(processed_image)
            
            # 6. In kết quả
            predicted_index = np.argmax(predictions[0]) 
            confidence = np.max(predictions[0]) * 100 
            predicted_class_name = idx_to_class[predicted_index]
            
            print("\n--- KẾT QUẢ DỰ ĐOÁN (MobileNetV2) ---")
            print(f"Ảnh: {image_path_to_test}")
            print(f"Lớp dự đoán: {predicted_class_name}")
            print(f"Độ tự tin (Confidence): {confidence:.2f}%")
            print("-------------------------------------\n")

except Exception as e:
    print(f"Lỗi trong quá trình dự đoán: {e}")
    print(f"Đảm bảo file '{MODEL_PATH_PREDICT}' và '{CLASS_INDICES_PATH}' tồn tại.")




