# 🧠 DỰ ÁN NGHIÊN CỨU PHÁT HIỆN HÌNH ẢNH GIẢ MẠO (DEEPFAKE) BẰNG HỌC SÂU

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11.9-blue" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.20.0-orange" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Accuracy-95%25-brightgreen" alt="Accuracy">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Status">
</p>

---

## 📋 Mục lục
1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Cấu hình hệ thống và chuẩn bị dữ liệu](#2-cấu-hình-hệ-thống-và-chuẩn-bị-dữ-liệu)
3. [Kiến trúc mô hình](#3-kiến-trúc-mô-hình)
4. [Quá trình training](#4-quá-trình-training)
5. [Kết quả và đánh giá](#5-kết-quả-và-đánh-giá)
6. [Đóng góp chính của nghiên cứu](#6-đóng-góp-chính-của-nghiên-cứu)
7. [Thách thức và giải pháp](#7-thách-thức-và-giải-pháp)
8. [Hướng phát triển tương lai](#8-hướng-phát-triển-tương-lai)
9. [Kết luận](#9-kết-luận)

---

## 1. Tổng quan dự án

Dự án tập trung xây dựng một hệ thống **phát hiện hình ảnh deepfake** sử dụng **học sâu (Deep Learning)** với kiến trúc **EfficientNetB0** làm backbone.

Dataset được sử dụng là **FaceForensics++** (trích xuất frame từ video), kết hợp **tăng cường dữ liệu truyền thống** và **GAN-based augmentation (DCGAN)** nhằm cải thiện khả năng tổng quát hóa của mô hình.

### 🎯 Mục tiêu chính
- Xây dựng hệ thống phát hiện deepfake tự động  
- Đạt độ chính xác cao trên tập test  
- Tối ưu để **chạy hoàn toàn trên CPU (không GPU)**  

---

## 2. Cấu hình hệ thống và chuẩn bị dữ liệu

### 2.1. Cấu hình hệ thống

| Thành phần | Thông số |
|-----------|----------|
| Hệ điều hành | Windows 10 |
| CPU | 6 cores vật lý |
| RAM | 15.78 GB |
| GPU | Không có |
| Python | 3.11.9 |
| TensorFlow | 2.20.0 |

---

### 2.2. Quy trình xử lý dữ liệu

#### a) Trích xuất khuôn mặt từ video
- Sử dụng **MTCNN** để phát hiện và cắt khuôn mặt  
- Xử lý **200 video thật** và **200 video giả**  
- Lấy **mỗi frame thứ 10** để giảm tải tính toán  

📊 **Kết quả:**
- 16,789 ảnh thật  
- 14,142 ảnh giả  

---

#### b) Chuẩn hóa kích thước ảnh
- Resize về **224 × 224 pixels**
- Sử dụng `cv2.INTER_AREA` để đảm bảo chất lượng

---

#### c) Tăng cường dữ liệu

##### 🔹 Traditional Augmentation (2,000 ảnh giả)
- Lật ngang (50%)
- Xoay (-15° → +15°)
- Điều chỉnh độ sáng (0.8 → 1.2)
- Zoom (0.85 → 1.0)

##### 🔹 DCGAN Augmentation
- DCGAN **64×64** → upscale lên 224×224
- DCGAN **224×224** (train trực tiếp)
- Tạo **2,000 ảnh giả từ mỗi mô hình**

---

## 3. Kiến trúc mô hình

### 3.1. EfficientNetB0 Architecture

<p align="center">
  <img src="https://github.com/user-attachments/assets/d80763ba-5d2f-40fb-bbd3-c01388804e8d" width="272" height="775">
</p>

#### 🔹 Backbone
- EfficientNetB0 (pre-trained ImageNet)
- Loại bỏ top layer
- Fine-tuning **50 layer cuối**

#### 🔹 Classification Head
- GlobalAveragePooling2D  
- Batch Normalization  
- Dropout (0.5)  
- Dense (512, ReLU)  
- Dropout (0.4)  
- Dense (1, Sigmoid)

---

### 3.2. Thông số kỹ thuật

| Thông số | Giá trị |
|--------|--------|
| Total Params | 4,711,076 (~18MB) |
| Trainable Params | 3,185,809 |
| Non-trainable Params | 1,525,267 |
| Input Shape | (224, 224, 3) |
| Output | Xác suất Fake |

---

## 4. Quá trình training

### 4.1. Chia dữ liệu

| Split | Số ảnh | Tỷ lệ |
|-----|-------|-------|
| Training | 26,344 | 80% |
| Validation | 3,293 | 10% |
| Testing | 3,294 | 10% |

---

### 4.2. Data Pipeline
- Random Horizontal Flip  
- Random Rotation (10%)  
- Random Zoom (10%)  
- Random Contrast (10%)

---

### 4.3. Training Configuration

| Tham số | Giá trị |
|------|--------|
| Batch Size | 8 |
| Epochs | 20 |
| Optimizer | Adam + Cosine Decay |
| Initial LR | 1e-4 |
| Loss | Binary Crossentropy |

**Callbacks**
- ModelCheckpoint  
- EarlyStopping (patience = 5)

---

## 5. Kết quả và đánh giá

### 5.1. Performance Metrics

| Metric | Giá trị |
|------|--------|
| Training Accuracy | 94.45% |
| Validation Accuracy | 94.23% |
| Best Val Accuracy | 94.32% |
| Test Accuracy | **95%** |
<img width="621" height="291" alt="image" src="https://github.com/user-attachments/assets/38f6efcc-2edf-44c3-848e-c3871df112e3" />

--- 


### 5.2. Confusion Matrix

| | Pred Fake | Pred Real |
|--|----------|-----------|
| Actual Fake | 92% | 8% |
| Actual Real | 3% | 97% |

---

## 6. Đóng góp chính của nghiên cứu

✅ Pipeline xử lý dữ liệu hoàn chỉnh  
✅ Kết hợp augmentation truyền thống & GAN  
✅ Training hiệu quả trên CPU  
✅ Mô hình nhẹ (~18MB) – accuracy cao  
✅ Khả năng tổng quát hóa tốt  

---

## 7. Thách thức và giải pháp

### 7.1. Thách thức
- Không GPU  
- Dataset mất cân bằng  
- Giới hạn RAM  

### 7.2. Giải pháp
- Batch size nhỏ  
- Data augmentation  
- tf.data + prefetch  
- EfficientNetB0  

---

## 8. Hướng phát triển tương lai

🔬 **Nghiên cứu**
- Vision Transformer
- ResNet variants
- Dataset đa nguồn

🚀 **Ứng dụng**
- Phát hiện deepfake video
- API real-time
- Tích hợp hệ thống bảo mật

📊 **Cải tiến**
- Ensemble models
- Explainable AI (Grad-CAM)
- Tối ưu inference

---

## 9. Kết luận

🎉 Mô hình đạt **95% accuracy**, hoạt động ổn định trên CPU  
💡 Kết hợp Transfer Learning + GAN cho hiệu quả cao  
🏗️ Pipeline hoàn chỉnh, dễ mở rộng  
🌍 Góp phần chống thông tin sai lệch và deepfake

---

<p align="center">
  <em>📊 Mô hình đạt 95% độ chính xác – Sẵn sàng cho ứng dụng thực tế</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-Python-3776AB.svg?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Powered%20by-TensorFlow-FF6F00.svg?style=for-the-badge&logo=tensorflow&logoColor=white">
  <img src="https://img.shields.io/badge/Research-Complete-success.svg?style=for-the-badge">
</p>
