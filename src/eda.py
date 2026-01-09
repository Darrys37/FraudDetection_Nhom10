import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

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
    top_4_features = feature_names[indices][:4]
    
    print(f"-> Top 4 đặc trưng quan trọng nhất: {list(top_4_features)}")

    # 2. Chuẩn bị dữ liệu vẽ
    df_plot = pd.DataFrame(X_test, columns=feature_names).copy()
    df_plot['is_fraud'] = y_test.values

    # 3. Vẽ biểu đồ
    fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # Tạo cửa sổ mới
    plt.suptitle('PHÂN TÍCH ĐẶC ĐIỂM GIAO DỊCH GIAN LẬN', fontsize=16)
    axes = axes.flatten()

    for i, col in enumerate(top_4_features):
        sns.kdeplot(data=df_plot[df_plot['is_fraud']==0], x=col, label='Normal', ax=axes[i], fill=True, color='blue', alpha=0.3)
        sns.kdeplot(data=df_plot[df_plot['is_fraud']==1], x=col, label='Fraud', ax=axes[i], fill=True, color='red', alpha=0.3)
        axes[i].set_title(f'Phân phối: {col}')
        axes[i].legend()
    
    plt.tight_layout()
