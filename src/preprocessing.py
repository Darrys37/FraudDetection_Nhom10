import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

def load_and_clean_data(filepath):
    """Đọc và làm sạch dữ liệu ban đầu"""
    print(f"--- [Preprocessing] Đang đọc file: {filepath} ---")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Không tìm thấy file tại {filepath}")

    # 1. Xóa các cột ID không cần thiết
    ids_to_drop = ['transaction_id', 'user_id', 'trans_num', 'unix_time']
    df.drop(columns=[c for c in ids_to_drop if c in df.columns], inplace=True, errors='ignore')

    # 2. Xử lý target
    if 'is_fraud' in df.columns:
        df = df.dropna(subset=['is_fraud'])
        df['is_fraud'] = df['is_fraud'].astype(int)

    # 3. Xử lý thời gian (nếu còn cột transaction_time chưa bị xóa)
    if 'transaction_time' in df.columns:
        df['transaction_time'] = pd.to_datetime(df['transaction_time'])
        df['transaction_hour'] = df['transaction_time'].dt.hour
        df['day_of_week'] = df['transaction_time'].dt.dayofweek
        df.drop(columns=['transaction_time'], inplace=True)

    # 4. Xóa thông tin cá nhân rác
    ignore_cols = ['first_name', 'last_name', 'email', 'dob', 'job', 'street', 'city', 'state', 'zip']
    df.drop(columns=[c for c in ignore_cols if c in df.columns], errors='ignore', inplace=True)
    
    return df

def prepare_data_for_model(df):
    """Mã hóa, Scale và Chia tập dữ liệu (kèm Undersampling)"""
    print("--- [Preprocessing] Đang chuẩn bị dữ liệu (Encoding & Splitting) ---")
    
    # 1. One-Hot Encoding
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    # 2. Tách X, y
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    # 3. Chia Train/Test (Trước khi Undersample để tránh data leakage)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 4. Scaling (Fit trên Train, Transform trên Test)
    scaler = MinMaxScaler()
    X_train_full = pd.DataFrame(scaler.fit_transform(X_train_full), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # 5. Undersampling (Chỉ trên tập Train)
    # Gom lại để resample
    train_data = pd.concat([X_train_full, y_train_full.reset_index(drop=True)], axis=1)
    
    not_fraud = train_data[train_data.is_fraud == 0]
    fraud = train_data[train_data.is_fraud == 1]

    # Tỷ lệ 1:10 (theo bài của bạn)
    n_fraud = len(fraud)
    n_not_fraud = n_fraud * 10 
    
    if len(not_fraud) > n_not_fraud:
        not_fraud_downsampled = resample(not_fraud, replace=False, n_samples=n_not_fraud, random_state=42)
    else:
        not_fraud_downsampled = not_fraud # Nếu không đủ dữ liệu thì giữ nguyên

    df_train_balanced = pd.concat([fraud, not_fraud_downsampled])
    
    # Tách lại X_train, y_train
    X_train = df_train_balanced.drop('is_fraud', axis=1)
    y_train = df_train_balanced['is_fraud']

    print(f"-> Shape Train sau Undersample: {X_train.shape}")
    print(f"-> Shape Test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, X.columns