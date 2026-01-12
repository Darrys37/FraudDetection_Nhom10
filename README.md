DO AN PHAT HIEN GIAN LAN GIAO DICH (FRAUD DETECTION)

1. Yeu cau moi truong
- Python 3.8 tro len.
- Dam bao cau truc thu muc dung yeu cau:
  + Du lieu 'transactions.csv' nam trong thu muc 'data/'
  + Source code nam trong thu muc 'src/'

2. Huong dan cai dat
Mo terminal tai thu muc goc cua project va chay lenh sau de cai dat cac thu vien can thiet:

pip install -r requirements.txt

3. Huong dan chay chuong trinh
Co the chay du an theo 2 cach duoi day:

Cach 1: Chay quy trinh Huan luyen & Danh gia (Terminal)
Lenh nay se chay tuan tu cac buoc: Xu ly du lieu -> EDA -> Train 3 Model (Decision Tree, Random Forest, XGBoost) -> Danh gia.

Lenh chay:
python src/main.py

Luu y: Trong qua trinh chay, neu cua so bieu do (EDA) hien len, hay dong cua so bieu do do lai de chuong trinh tiep tuc chay sang buoc huan luyen mo hinh.

Cach 2: Chay Web App (Dashboard)
Su dung Streamlit de xem giao dien so sanh mo hinh va demo du doan:

Lenh chay:
streamlit run src/app.py
