import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 1. KHỞI TẠO

app = Flask(__name__)

MODEL_PATH = "best_mobilenet.h5"
CAT_TO_NAME_PATH = "cat_to_name.json"
TRAIN_DIR = "flower_data_split/train"

# Load model
model = load_model(MODEL_PATH)
print(" Model loaded successfully")

# Load file cat_to_name.json
with open(CAT_TO_NAME_PATH, "r") as f:
    cat_to_name = json.load(f)

# Tạo generator tạm để lấy class_indices
datagen = ImageDataGenerator(rescale=1./255)
temp_gen = datagen.flow_from_directory(
    TRAIN_DIR, target_size=(224, 224), batch_size=1, shuffle=False
)
class_indices = temp_gen.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}
print(f" Found {len(idx_to_class)} classes in training directory.")


# 2. HÀM DỰ ĐOÁN

def predict_flower(img_path):
    IMG_SIZE = (224, 224)
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_idx = int(np.argmax(preds[0]))          # ép kiểu int thuần
    conf = float(np.max(preds[0]))               # độ tin cậy

    # Map index → class id thật
    class_id = idx_to_class.get(pred_idx, None)
    if class_id is None:
        class_id = str(pred_idx + 1)

    # Chuẩn hóa "001" -thanh"1", và đảm bảo dạng chuỗi
    try:
        class_id = str(int(class_id))
    except Exception:
        class_id = str(class_id)

    # Map class_id thanh  tên hoa trong JSON
    flower_name = cat_to_name.get(str(class_id), f"Lớp {class_id}")
    flower_name = flower_name.title() if isinstance(flower_name, str) else str(flower_name)

    return flower_name, conf


# 3. ROUTES CHÍNH

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Không có file upload!"})

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "File không hợp lệ!"})

        os.makedirs("uploads", exist_ok=True)
        img_path = os.path.join("uploads", file.filename)
        file.save(img_path)

        flower_name, conf = predict_flower(img_path)

        print(f"[DEBUG] {file.filename} → {flower_name} ({conf:.2%})")

        return jsonify({
            "flower_name": flower_name,
            "confidence": f"{conf*100:.2f}%"
        })

    except Exception as e:
        print(f"❌ ERROR: {repr(e)}")
        return jsonify({"error": str(e)})

 
#4. RUN

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
