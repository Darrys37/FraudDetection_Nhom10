import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

def load_and_clean_data(filepath):
    """
    Đọc và làm sạch dữ liệu ban đầu.
    - Xóa cột ID, thông tin cá nhân.
    - Xử lý thời gian (tạo cột hour, day_of_week).
    """
    print(f"--- [Preprocessing] Đang đọc file: {filepath} ---")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Không tìm thấy file tại {filepath}")

    # 1. Xóa các cột ID không cần thiết (Giống code cũ)
    ids_to_drop = ['transaction_id', 'user_id', 'trans_num', 'unix_time']
    df.drop(columns=[c for c in ids_to_drop if c in df.columns], inplace=True, errors='ignore')

    # 2. Xóa các cột thông tin cá nhân rác (Giống code cũ)
    ignore_cols = ['first_name', 'last_name', 'email', 'dob', 'job', 'street', 'city', 'state', 'zip']
    df.drop(columns=[c for c in ignore_cols if c in df.columns], errors='ignore', inplace=True)

    # 3. Xử lý target
    if 'is_fraud' in df.columns:
        df = df.dropna(subset=['is_fraud'])
        df['is_fraud'] = df['is_fraud'].astype(int)

    # 4. Xử lý thiếu dữ liệu (Fill NA)
    cat_cols = df.select_dtypes(include=['object']).columns
    num_cols = df.select_dtypes(include=['number']).columns
    df[cat_cols] = df[cat_cols].fillna('Unknown')
    df[num_cols] = df[num_cols].fillna(0)

    # 5. Xử lý thời gian [QUAN TRỌNG: SỬA LỖI MISSING 'HOUR']
    if 'transaction_time' in df.columns:
        df['transaction_time'] = pd.to_datetime(df['transaction_time'])
        
        # Tạo cột 'transaction_hour' (cho Model dùng)
        df['transaction_hour'] = df['transaction_time'].dt.hour
        
        # [FIX] Tạo thêm cột 'hour' (cho EDA dùng để vẽ biểu đồ)
        df['hour'] = df['transaction_time'].dt.hour 
        
        # Tạo cột ngày trong tuần
        df['day_of_week'] = df['transaction_time'].dt.dayofweek
        
        # Xóa cột thời gian gốc sau khi đã tách
        df.drop(columns=['transaction_time'], inplace=True)
        
        print("- Đã xử lý thời gian: Tạo cột 'hour', 'transaction_hour', 'day_of_week'")

    return df

def prepare_data_for_model(df):
    """
    Chuẩn bị dữ liệu để đưa vào huấn luyện (Encoding -> Split -> Scale -> Undersample)
    """
    print("--- [Preprocessing] Đang chuẩn bị dữ liệu cho Model ---")
    
    # 1. One-Hot Encoding (Chuyển chữ thành số)
    # Lấy danh sách các cột object (trừ is_fraud)
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"- Đang One-Hot Encoding các cột: {list(categorical_cols)}")
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # 2. Tách Feature (X) và Target (y)
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    # Lưu lại tên các đặc trưng
    feature_names = X.columns

    # 3. Chia Train/Test (Trước khi Undersample để tránh data leakage)
    # Stratify=y để đảm bảo tỷ lệ fraud ở 2 tập đều nhau
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 4. Scaling (Fit trên Train, Transform trên Test)
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

    # 5. Undersampling (Chỉ thực hiện trên tập Train)
    print("- Đang thực hiện Undersampling trên tập Train...")
    
    # Gom lại tạm thời để resample
    train_data = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    
    not_fraud = train_data[train_data.is_fraud == 0]
    fraud = train_data[train_data.is_fraud == 1]

    # Tỷ lệ 1:10 (1 Fraud : 10 Normal) - Giống logic bài của bạn
    n_fraud = len(fraud)
    n_not_fraud = n_fraud * 10 
    
    # Nếu số lượng normal nhiều hơn mức cần thiết thì cắt bớt
    if len(not_fraud) > n_not_fraud:
        not_fraud_downsampled = resample(not_fraud,
                                         replace=False,
                                         n_samples=n_not_fraud,
                                         random_state=42)
    else:
        not_fraud_downsampled = not_fraud

    # Gộp lại thành tập Train cân bằng
    train_balanced = pd.concat([fraud, not_fraud_downsampled])
    
    # Trộn ngẫu nhiên (Shuffle)
    train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Tách lại X_train, y_train
    X_train_final = train_balanced.drop('is_fraud', axis=1)
    y_train_final = train_balanced['is_fraud']

    print(f"-> Kích thước Train sau Undersampling: {X_train_final.shape}")
    print(f"-> Kích thước Test (giữ nguyên): {X_test.shape}")

    return X_train_final, X_test, y_train_final, y_test, feature_names