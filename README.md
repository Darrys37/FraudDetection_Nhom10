=============================================================================
PHÁT HIỆN GIAN LẬN GIAO DỊCH (E-Commerce Fraud Detection Dataset) - NHÓM 10
=============================================================================

Dự án áp dụng các kỹ thuật Học máy (Machine Learning) để phân loại và phát 
hiện các giao dịch thẻ tín dụng gian lận. Hệ thống giải quyết bài toán mất 
cân bằng dữ liệu nghiêm trọng, tích hợp quy trình xử lý tự động và triển 
khai ứng dụng Web thực tế.

-----------------------------------------------------------------------------
1. CÀI ĐẶT MÔI TRƯỜNG
-----------------------------------------------------------------------------
- Yêu cầu hệ thống: Python 3.8 trở lên.

- Bước 1: Clone dự án hoặc tải source code về máy.
- Bước 2: Cài đặt các thư viện phụ thuộc bằng lệnh sau:
  
  pip install -r requirements.txt

-----------------------------------------------------------------------------
2. HƯỚNG DẪN CHẠY CHƯƠNG TRÌNH
-----------------------------------------------------------------------------
CÁCH 1: Chạy quy trình Huấn luyện & Đánh giá (Backend)
- Chạy lệnh sau để thực hiện toàn bộ pipeline:
  (Load dữ liệu -> Xử lý -> Train 3 Models -> Xuất báo cáo so sánh)

  python src/main.py

- Kết quả:
  + Hiển thị các chỉ số Recall, F1-Score trên Terminal.
  + Lưu các model đã train vào thư mục models/ (file .pkl).
  + Tự động lưu file dữ liệu đã làm sạch vào data/.

CÁCH 2: Chạy Giao diện Web (Frontend - Streamlit)
- Để mở ứng dụng Demo kiểm tra gian lận (có biểu đồ trực quan):

  python -m streamlit run app.py

- Hệ thống sẽ mở trình duyệt tại địa chỉ: http://localhost:8501

-----------------------------------------------------------------------------
3. CẤU TRÚC THƯ MỤC SOURCE CODE
-----------------------------------------------------------------------------
FraudDetection_Nhom10/
|-- data/
|   |-- transactions.csv               # Dữ liệu giao dịch gốc
|
|-- models/                            # Chứa các file model (.pkl) sau khi train
|
|-- src/                               # MÃ NGUỒN CHÍNH
|   |-- eda.py                         # Phân tích khám phá dữ liệu
|   |-- preprocessing.py               # Làm sạch & Cân bằng dữ liệu (Undersampling)
|   |-- feature_engineering.py         # Tạo đặc trưng mới (Feature Engineering)
|   |-- evaluation.py                  # Các hàm đánh giá & vẽ biểu đồ
|   |-- model_DecisionTree_TranTatPhat.py   # Model Decision Tree
|   |-- model_RandomForest_DoXuanHuong.py   # Model Random Forest
|   |-- model_XGBoost_NguyenHuynhAnhTuan.py # Model XGBoost
|   |-- main.py                        # File điều phối luồng chạy chính
|
|-- app.py                             # Giao diện Web Streamlit
|-- requirements.txt                   # Danh sách thư viện
|-- README.txt                         # Hướng dẫn sử dụng
|-- N10_report.pdf                     # Báo cáo project
|-- temp_uploaded.csv                  # File tạm sinh ra khi upload dữ liệu trên Web
