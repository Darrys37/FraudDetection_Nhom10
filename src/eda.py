import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# =============================================================================
# PHẦN 1: MODULE 2.5 - EXPLORATORY DATA ANALYSIS (Theo yêu cầu của Tuấn)
# =============================================================================

def run_eda_analysis(df):
    """
    Hàm thực hiện phân tích dữ liệu khám phá (EDA) tổng quan.
    Bao gồm: Class Distribution, Hourly Analysis, Correlation, Channel/Category, Amount.
    """
    print(f"{'='*15} BẮT ĐẦU PHÂN TÍCH EDA {'='*15}")
    
    # 0. PHÂN PHỐI NHÃN (CLASS DISTRIBUTION)
    print("\n0. Đang vẽ phân phối nhãn (Class Distribution)...")
    plt.figure(figsize=(6, 4))
    sns.countplot(x='is_fraud', data=df, palette=['skyblue', 'red'])
    plt.title('Phân phối các lớp (0: Normal, 1: Fraud)')
    plt.xlabel('Trạng thái')
    plt.ylabel('Số lượng')
    # plt.show()

    # Tính tỷ lệ cụ thể để in ra cho rõ
    fraud_count = df['is_fraud'].sum()
    total_count = len(df)
    print(f"-> Số lượng Gian lận: {fraud_count} / {total_count}")
    print(f"-> Tỷ lệ Gian lận: {fraud_count/total_count:.2%}")

    # 1. PHÂN TÍCH THEO GIỜ
    print("\n1. Đang vẽ biểu đồ theo khung giờ (Hourly)...")
    if 'hour' in df.columns:
        fraud_by_hour = df.groupby('hour')['is_fraud'].mean()
        tx_by_hour = df.groupby('hour')['is_fraud'].count()

        # Biểu đồ 1: Số lượng giao dịch
        plt.figure(figsize=(10, 4))
        tx_by_hour.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Tổng số lượng giao dịch theo Giờ')
        plt.ylabel('Số lượng (Count)')
        plt.xlabel('Giờ trong ngày')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # plt.show()

        # Biểu đồ 2: Tỷ lệ gian lận
        plt.figure(figsize=(10, 4))
        fraud_by_hour.plot(kind='bar', color='salmon', edgecolor='black')
        plt.title('Tỷ lệ Gian lận theo Giờ (Fraud Rate)')
        plt.ylabel('Tỷ lệ gian lận')
        plt.xlabel('Giờ trong ngày')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # plt.show()

        print("Top 5 khung giờ có tỷ lệ gian lận cao nhất:")
        print(fraud_by_hour.sort_values(ascending=False).head(5))
    else:
        print("Cảnh báo: Không tìm thấy cột 'hour'.")

    # 2. MA TRẬN TƯƠNG QUAN
    print("\n2. Đang vẽ ma trận tương quan (Numeric Features)...")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Loại bỏ các cột merchant, country, id để biểu đồ đỡ rối
    cols_to_plot = [c for c in num_cols if 'merchant_' not in c and 'country_' not in c and 'id' not in c]
    
    if 'is_fraud' not in cols_to_plot and 'is_fraud' in df.columns: 
        cols_to_plot.append('is_fraud')
        
    corr = df[cols_to_plot].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix (Các đặc trưng số chính)')
    plt.tight_layout()
    # plt.show()

    print("Top 10 đặc trưng tương quan mạnh nhất với 'is_fraud':")
    if 'is_fraud' in corr.columns:
        print(corr['is_fraud'].abs().sort_values(ascending=False).head(10))

    # 3. PHÂN TÍCH CHANNEL & CATEGORY
    print("\n3. Đang vẽ biểu đồ theo Channel & Merchant Category...")
    
    df_viz = df.copy()

    # Xử lý Merchant Category
    merch_cols = [c for c in df.columns if 'merchant_category_' in c]
    col_cat = None
    if merch_cols:
        # Nếu đã one-hot encode, gộp lại để vẽ
        df_viz['merchant_category_label'] = df_viz[merch_cols].idxmax(axis=1).apply(lambda x: x.replace('merchant_category_', ''))
        col_cat = 'merchant_category_label'
    elif 'merchant_category' in df.columns:
        col_cat = 'merchant_category'

    if col_cat:
        fraud_by_category = df_viz.groupby(col_cat)['is_fraud'].mean().sort_values(ascending=False)
        plt.figure(figsize=(10, 5))
        fraud_by_category.plot(kind='bar', color='orange', edgecolor='black')
        plt.title('Tỷ lệ Gian lận theo Loại hình bán hàng (Merchant Category)')
        plt.ylabel('Tỷ lệ gian lận')
        plt.xticks(rotation=45)
        # plt.show()

    # Xử lý Channel
    if 'channel' in df_viz.columns:
        if pd.api.types.is_numeric_dtype(df_viz['channel']):
             df_viz['channel_label'] = df_viz['channel'].map({1: 'Web', 0: 'App'})
        else:
             df_viz['channel_label'] = df_viz['channel']
        
        fraud_by_channel = df_viz.groupby('channel_label')['is_fraud'].mean().sort_values(ascending=False)
        plt.figure(figsize=(6, 4))
        fraud_by_channel.plot(kind='bar', color='purple', edgecolor='black')
        plt.title('Tỷ lệ Gian lận theo Kênh (Channel)')
        plt.ylabel('Tỷ lệ gian lận')
        # plt.show()

    # 4. PHÂN TÍCH SỐ TIỀN
    print("\n4. Đang vẽ phân phối số tiền giao dịch (Amount)...")
    if 'amount' in df.columns:
        amount = df['amount'].astype(float)
        amount_pos = amount[amount > 0] 
        log_amount = np.log(amount_pos)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.hist(amount_pos, bins=60, color='green', alpha=0.7)
        ax1.set_title('Phân phối số tiền (Gốc)')
        ax1.set_xlabel('Amount ($)')
        ax1.set_ylabel('Tần suất')

        ax2.hist(log_amount, bins=60, color='teal', alpha=0.7)
        ax2.set_title('Phân phối số tiền (Log Transform)')
        ax2.set_xlabel('Log(Amount)')
        ax2.set_ylabel('Tần suất')
        
        # plt.show()

        print("Thống kê số tiền:")
        stats = {
            "amount_mean": float(amount_pos.mean()),
            "amount_std": float(amount_pos.std(ddof=1)),
        }
        print(stats)


# =============================================================================
# PHẦN 2: ANALYZE HIDDEN PATTERNS (Code hiện tại của Project)
# =============================================================================

def analyze_hidden_patterns(model, X_test, y_test, feature_names):
    """Tìm hidden patterns dựa trên Feature Importance"""
    print(f"\n{'='*20} [EDA] PHÂN TÍCH HIDDEN PATTERNS {'='*20}")
    
    # 1. Lấy Feature Importance
    try:
        importances = model.feature_importances_
    except AttributeError:
        print("Mô hình này không hỗ trợ feature_importances_")
        return

    indices = np.argsort(importances)[::-1]
    # Đảm bảo không lấy quá số lượng feature hiện có
    top_n = min(4, len(feature_names))
    top_features = feature_names[indices][:top_n]
    
    print(f"-> Top {top_n} đặc trưng quan trọng nhất: {list(top_features)}")

    # 2. Chuẩn bị dữ liệu vẽ
    # Lưu ý: X_test cần là numpy array hoặc dataframe tương thích
    df_plot = pd.DataFrame(X_test, columns=feature_names).copy()
    df_plot['is_fraud'] = y_test.values

    # 3. Vẽ biểu đồ
    fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # Tạo cửa sổ mới
    plt.suptitle('PHÂN TÍCH ĐẶC ĐIỂM GIAO DỊCH GIAN LẬN', fontsize=16)
    axes = axes.flatten()

    for i, col in enumerate(top_features):
        if i >= len(axes): break # Tránh lỗi index nếu có nhiều hơn 4 feature
        
        # Vẽ phân phối cho lớp Normal
        sns.kdeplot(data=df_plot[df_plot['is_fraud']==0], x=col, label='Normal', 
                    ax=axes[i], fill=True, color='blue', alpha=0.3)
        # Vẽ phân phối cho lớp Fraud
        sns.kdeplot(data=df_plot[df_plot['is_fraud']==1], x=col, label='Fraud', 
                    ax=axes[i], fill=True, color='red', alpha=0.3)
        
        axes[i].set_title(f'Phân phối: {col}')
        axes[i].legend()
    
    plt.tight_layout()
    # plt.show()

