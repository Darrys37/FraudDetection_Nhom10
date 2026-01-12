PHÁT HIỆN GIAN LẬN GIAO DỊCH (E-Commerce Fraud Detection Dataset) - NHÓM 10

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
  
  python -m pip install -r requirements.txt

-----------------------------------------------------------------------------
2. HƯỚNG DẪN CHẠY CHƯƠNG TRÌNH
-----------------------------------------------------------------------------
CÁCH 1: Chạy quy trình Huấn luyện & Đánh giá (Backend)
- Chạy lệnh sau để thực hiện toàn bộ pipeline:
  (Load dữ liệu -> Xử lý -> Train 3 Models -> Xuất báo cáo so sánh)

  python src/main.py
  
CÁCH 2: Chạy Giao diện Web (Frontend - Streamlit)
- Để mở ứng dụng Demo kiểm tra gian lận (có biểu đồ trực quan):

  python -m streamlit run app.py

- Hệ thống sẽ mở trình duyệt tại địa chỉ: http://localhost:8501

