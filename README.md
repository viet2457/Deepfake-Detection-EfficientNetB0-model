DỰ ÁN NGHIÊN CỨU PHÁT HIỆN HÌNH ẢNH GIẢ MẠO (DEEPFAKE) BẰNG HỌC SÂU
<p align="center"> <img src="https://img.shields.io/badge/Python-3.11.9-blue" alt="Python"> <img src="https://img.shields.io/badge/TensorFlow-2.20.0-orange" alt="TensorFlow"> <img src="https://img.shields.io/badge/Accuracy-95%25-brightgreen" alt="Accuracy"> <img src="https://img.shields.io/badge/Status-Completed-success" alt="Status"> </p>
📋 Mục lục
1. Tổng quan dự án

2. Cấu hình hệ thống và chuẩn bị dữ liệu

2.1. Cấu hình hệ thống

2.2. Quy trình xử lý dữ liệu

3. Kiến trúc mô hình

3.1. EfficientNetB0 Architecture

3.2. Thông số kỹ thuật

4. Quá trình training

4.1. Chia dữ liệu

4.2. Data Pipeline

4.3. Training Configuration

5. Kết quả và đánh giá

5.1. Performance Metrics

5.2. Confusion Matrix

6. Đóng góp chính của nghiên cứu

7. Thách thức và giải pháp

7.1. Thách thức

7.2. Giải pháp

8. Hướng phát triển tương lai

9. Kết luận

1. Tổng quan dự án
Dự án này tập trung vào việc phát triển một mô hình học sâu để phát hiện hình ảnh deepfake, sử dụng kiến trúc EfficientNetB0 làm backbone. Nghiên cứu sử dụng tập dữ liệu FaceForensics++ từ Kaggle và áp dụng các kỹ thuật tăng cường dữ liệu cùng với mô hình GAN để cải thiện hiệu suất phát hiện.

🎯 Mục tiêu chính:
Xây dựng hệ thống phát hiện deepfake tự động

Đạt độ chính xác cao trên tập dữ liệu thử nghiệm

Tối ưu hóa cho hệ thống không có GPU

2. Cấu hình hệ thống và chuẩn bị dữ liệu
2.1. Cấu hình hệ thống
Thành phần	Thông số kỹ thuật
Hệ điều hành	Windows 10
RAM	15.78 GB
CPU	6 cores vật lý
GPU	Không có GPU (chạy hoàn toàn trên CPU)
TensorFlow	Version 2.20.0
Python	Version 3.11.9
2.2. Quy trình xử lý dữ liệu
a) Trích xuất khuôn mặt từ video
python
- Sử dụng MTCNN để tự động phát hiện và cắt khuôn mặt từ video
- Xử lý 200 video thật và 200 video giả mạo
- Cài đặt: Xử lý mỗi khung hình thứ 10 để giảm tải
- Kết quả: 16,789 ảnh thật và 14,142 ảnh giả
b) Chuẩn hóa kích thước ảnh
python
- Resize tất cả ảnh về kích thước 224x224 pixel
- Sử dụng interpolation cv2.INTER_AREA để đảm bảo chất lượng
c) Tăng cường dữ liệu
Traditional Augmentation (Tạo thêm 2,000 ảnh giả):

Lật ngang (50% xác suất)

Xoay nhẹ (-15 đến +15 độ)

Điều chỉnh độ sáng (0.8-1.2)

Zoom nhẹ (0.85-1.0)

DCGAN Augmentation (Huấn luyện DCGAN để tạo ảnh giả mới):

Phiên bản 64x64: Training nhanh, sau đó upscale lên 224x224

Phiên bản 224x224: Training trực tiếp ở độ phân giải mục tiêu

Tạo 2,000 ảnh giả mới từ mỗi mô hình

3. Kiến trúc mô hình
3.1. EfficientNetB0 Architecture








Backbone (Feature Extractor):
Base model: EfficientNetB0 (không bao gồm top layer)

Transfer Learning: Sử dụng pre-trained weights từ ImageNet

Fine-tuning: Chỉ unfreeze 50 layer cuối để tinh chỉnh

Classification Head:
Global Average Pooling 2D: Chuyển đổi feature maps 7x7x1280 thành vector 1280

Batch Normalization: Ổn định quá trình training

Dropout (0.5): Ngăn chặn overfitting

Dense Layer (512 units, ReLU): Học các đặc trưng phức tạp

Dropout (0.4): Thêm regularization

Output Layer (1 unit, Sigmoid): Phân loại nhị phân (0: Real, 1: Fake)

3.2. Thông số kỹ thuật
Thông số	Giá trị
Total Parameters	4,711,076 (17.97 MB)
Trainable Parameters	3,185,809 (12.15 MB)
Non-trainable Parameters	1,525,267 (5.82 MB)
Input Shape	(224, 224, 3)
Output Shape	(1,) - Xác suất giả mạo
4. Quá trình training
4.1. Chia dữ liệu
Tổng số ảnh: 32,931 ảnh

16,789 ảnh thật

16,142 ảnh giả

Tỷ lệ chia (80/10/10):

Split	Số lượng ảnh	Tỷ lệ
Training	26,344	80%
Validation	3,293	10%
Testing	3,294	10%
4.2. Data Pipeline
Data Augmentation trong training:

Random horizontal flip

Random rotation (10%)

Random zoom (10%)

Random contrast (10%)

4.3. Training Configuration
Tham số	Giá trị
Batch Size	8 (tối ưu cho RAM)
Epochs	20
Optimizer	Adam với Cosine Decay Learning Rate
Initial learning rate	1e-4
Alpha	0.01
Loss Function	Binary Crossentropy
Callbacks:

ModelCheckpoint: Lưu model tốt nhất dựa trên val_accuracy

EarlyStopping: Patience = 5 epochs

5. Kết quả và đánh giá
5.1. Performance Metrics
Training Progress:
Metric	Giá trị
Final Training Accuracy	94.45%
Final Validation Accuracy	94.23%
Best Validation Accuracy	94.32% (Epoch 19)
Lowest Validation Loss	0.1140 (Epoch 17)
Testing Results (trên 3,091 ảnh):
text
              precision    recall  f1-score   support

        Fake       0.97      0.92      0.95      1679
        Real       0.92      0.97      0.94      1412

    accuracy                           0.95      3091
   macro avg       0.94      0.95      0.95      3091
weighted avg       0.95      0.95      0.95      3091
5.2. Confusion Matrix
Predicted Fake	Predicted Real
Actual Fake	92%	8%
Actual Real	3%	97%
Kết quả chi tiết:

True Positive (Fake correctly identified): 92%

True Negative (Real correctly identified): 97%

Overall Accuracy: 95%

6. Đóng góp chính của nghiên cứu
✅ Xử lý dữ liệu toàn diện: Tự động trích xuất khuôn mặt, chuẩn hóa và tăng cường dữ liệu

✅ Kết hợp nhiều phương pháp augmentation: Sử dụng cả traditional augmentation và GAN-based augmentation

✅ Tối ưu hóa training trên CPU: Điều chỉnh batch size và pipeline để chạy hiệu quả trên hệ thống không có GPU

✅ Hiệu suất cao: Đạt 95% accuracy trên tập test với mô hình nhẹ (18MB)

✅ Khả năng tổng quát hóa tốt: Precision và recall cân bằng giữa hai lớp

7. Thách thức và giải pháp
7.1. Thách thức
Thách thức	Mô tả
Không có GPU	Training chậm trên CPU
Mất cân bằng dữ liệu	Số lượng ảnh thật nhiều hơn ảnh giả
Memory constraints	RAM hạn chế (15.78 GB)
7.2. Giải pháp
Giải pháp	Hiệu quả
Tối ưu batch size	Sử dụng batch size nhỏ (8) phù hợp với RAM
Data augmentation	Tạo thêm ảnh giả để cân bằng dataset
Efficient data pipeline	Sử dụng tf.data.Dataset với prefetch để tối ưu I/O
Model optimization	Chọn EfficientNetB0 - mô hình nhẹ nhưng hiệu quả
8. Hướng phát triển tương lai
🔬 Nghiên cứu tiếp theo:
Nâng cấp phần cứng: Sử dụng GPU để training nhanh hơn

Thử nghiệm với các kiến trúc khác: Vision Transformers, ResNet variants

Mở rộng dataset: Sử dụng thêm các dataset deepfake khác

🚀 Ứng dụng thực tế:
Phát triển cho video: Mở rộng từ phát hiện ảnh sang phát hiện video

Triển khai thực tế: Xây dựng API hoặc ứng dụng cho kiểm tra real-time

Tích hợp hệ thống: Kết hợp với các hệ thống bảo mật hiện có

📊 Cải thiện hiệu suất:
Ensemble models: Kết hợp nhiều mô hình để tăng độ chính xác

Explainable AI: Phát triển tính năng giải thích quyết định của mô hình

Real-time optimization: Tối ưu hóa cho inference tốc độ cao

9. Kết luận
🎉 Thành tựu đạt được:
Nghiên cứu đã thành công trong việc phát triển một mô hình phát hiện deepfake hiệu quả sử dụng EfficientNetB0. Mô hình đạt được độ chính xác 95% trên tập test, chứng minh khả năng phân biệt tốt giữa ảnh thật và ảnh giả mạo.

💡 Phương pháp đột phá:
Phương pháp kết hợp giữa transfer learning, data augmentation truyền thống và GAN-based augmentation đã mang lại kết quả ấn tượng ngay cả trong điều kiện phần cứng hạn chế.

🏗️ Đóng góp chính:
Dự án cung cấp một pipeline hoàn chỉnh từ xử lý dữ liệu đến training và evaluation, có thể dễ dàng tái sử dụng và mở rộng cho các nghiên cứu tương tự trong tương lai.

🌟 Tác động:
Nghiên cứu này đóng góp vào việc phát triển các công cụ phát hiện deepfake tự động, hỗ trợ các nỗ lực chống lại thông tin sai lệch và bảo vệ tính xác thực của nội dung kỹ thuật số.

<p align="center"> <em>📊 Mô hình đạt 95% độ chính xác - Sẵn sàng cho ứng dụng thực tế</em> </p><p align="center"> <img src="https://img.shields.io/badge/Made%20with-Python-3776AB.svg?style=for-the-badge&logo=python&logoColor=white" alt="Made with Python"> <img src="https://img.shields.io/badge/Powered%20by-TensorFlow-FF6F00.svg?style=for-the-badge&logo=tensorflow&logoColor=white" alt="Powered by TensorFlow"> <img src="https://img.shields.io/badge/Research-Complete-success.svg?style=for-the-badge" alt="Research Complete"> </p>
