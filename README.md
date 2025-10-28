Tiền xử lý: Tự động phân chia bộ dữ liệu 8000+ ảnh thành 3 tập train/validation/test (tỷ lệ 80-10-10).
 Tăng cường dữ liệu (Data Augmentation):** Áp dụng các phép xoay, lật, zoom... để làm giàu dữ liệu tra
 Huấn luyện 3 mô hình:**
    1.  CNN Tùy chỉnh (Custom CNN):** Một mô hình CNN 4 lớp được xây dựng từ đầu.
    2.  MobileNetV2 (Transfer Learning):** Sử dụng Kỹ thuật Học chuyển tiếp.
    3.  EfficientNetB0 (Transfer Learning):** Sử dụng Kỹ thuật Học chuyển tiếp.
*Đánh giá chi tiết:Tự động đánh giá cả 3 mô hình trên tập test, tạo báo cáo (Precision, Recall, F1-Score), vẽ ma trận nhầm lẫn, và liệt kê top 10 lớp chính xác/nhầm lẫn nhất.
Dự đoán ảnh đơn: Chạy dự đoán trên một ảnh ngẫu nhiên từ tập test để kiểm tra mô hình.

để chạy website, cần có thư viện flask
và gõ lệnh python app.py
chuyển file flower_data_split vào file web
đưa file .h5 của từng mô hình vào file web
có thể thay mô hình ở đoạn  MODEL_PATH = ".....h5" bằng các mô hình khác 

1. Clone repo:
   
   git clone https://github.com/vinhquocle/Flower_Classification.git
   cd Flower_Classification

2 cài môi trường 
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
python src/main.py
cd web
python app.py
Truy cập http://127.0.0.1:5000 để xem giao diện.

project/
├── README.md
├── Report.pdf
├── requirements.txt
├── src/
│ ├── preprocessing.py
│ ├── eda.py
│ ├── feature_engineering.py
│ ├── model_cnn.py
│ ├── model_mobileNetV2.py
│ ├── model_EfficientNetB0.py
│ ├── evaluation.py
│ ├── main.py
├── web/
│ ├── app.py
│ ├── templates/index.html
│ |__ .....h5
| |__ flower_data_split
│ └── cat_to_name.json

link dataset:https://drive.google.com/file/d/1hEytojL8-yMHRfIh74va7YkmN-qF80nW/view?usp=sharing
link bestmodel:https://drive.google.com/file/d/1bDd7dLWYwnFeaFH6IjQRd-z7bPaHL0HI/view?usp=sharing
