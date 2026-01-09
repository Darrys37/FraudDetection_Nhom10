import pandas as pd
import numpy as np

def create_new_features(df):
    """Tạo các cột đặc trưng mới"""
    print("--- [Feature Engineering] Đang tạo đặc trưng mới ---")
    
    # 1. Security Score
    security_cols = ['avs_match', 'cvv_result', 'three_ds_flag']
    available_sec = [c for c in security_cols if c in df.columns]
    if available_sec:
        # Lưu ý: cần map về số trước nếu chưa phải số
        df['security_score'] = df[available_sec].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)

    # 2. Amount Ratio
    if 'avg_amount_user' in df.columns and 'amount' in df.columns:
        df['amount_ratio'] = np.where(df['avg_amount_user'] > 0, df['amount'] / df['avg_amount_user'], 0)

    # 3. Geo Mismatch (Lệch quốc gia)
    if 'country' in df.columns and 'bin_country' in df.columns:
        c_clean = df['country'].astype(str).str.lower().str.strip()
        b_clean = df['bin_country'].astype(str).str.lower().str.strip()
        df['geo_mismatch'] = (c_clean != b_clean).astype(int)

    # 4. Weird Grocery Ship (Tạp hóa nhưng ship xa)
    if 'merchant_category' in df.columns and 'shipping_distance_km' in df.columns:
        df['weird_grocery_ship'] = ((df['merchant_category'] == 'grocery') & 
                                    (df['shipping_distance_km'] > 50)).astype(int)

    # 5. Distance per Dollar
    if 'shipping_distance_km' in df.columns and 'amount' in df.columns:
         df['distance_per_dollar'] = np.where(df['amount'] > 0, df['shipping_distance_km'] / df['amount'], 0)

    print(f"-> Đã thêm features. Kích thước hiện tại: {df.shape}")
    return df