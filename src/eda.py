import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st 

# =============================================================================
# MODULE 2.5 - EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

def run_eda_analysis(df, enable_streamlit=False):
    """
    Hàm thực hiện phân tích dữ liệu khám phá (EDA).
    """
    
    # Hàm con hỗ trợ hiển thị
    def show_fig(fig):
        if enable_streamlit:
            st.pyplot(fig)
            plt.close(fig) # Quan trọng: Xóa hình khỏi bộ nhớ sau khi vẽ
        else:
            # plt.show() # Tắt show console
            pass

    def show_text(text, header=False):
        if enable_streamlit:
            if header: st.subheader(text)
            else: st.write(text)
        else:
            print(text)

    # --- BẮT ĐẦU ---
    if enable_streamlit:
        st.markdown("### 1. Phân phối nhãn (Class Distribution)")
    
    # 0. PHÂN PHỐI NHÃN
    # Dùng subplots để tạo khung hình riêng biệt hoàn toàn
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.countplot(x='is_fraud', data=df, palette=['skyblue', 'red'], ax=ax1)
    ax1.set_title('Phân phối các lớp (0: Normal, 1: Fraud)')
    show_fig(fig1)

    fraud_count = df['is_fraud'].sum()
    total = len(df)
    show_text(f"-> Số lượng Gian lận: {fraud_count} / {total} ({fraud_count/total:.2%})")

    # 1. PHÂN TÍCH THEO GIỜ
    if enable_streamlit: st.markdown("### 2. Phân tích theo khung giờ")
    
    if 'hour' in df.columns:
        fraud_by_hour = df.groupby('hour')['is_fraud'].mean()
        tx_by_hour = df.groupby('hour')['is_fraud'].count()

        # Biểu đồ 1: Số lượng
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        tx_by_hour.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax2)
        ax2.set_title('Tổng số lượng giao dịch theo Giờ')
        show_fig(fig2)

        # Biểu đồ 2: Tỷ lệ gian lận
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        fraud_by_hour.plot(kind='bar', color='salmon', edgecolor='black', ax=ax3)
        ax3.set_title('Tỷ lệ Gian lận theo Giờ')
        show_fig(fig3)
    else:
        show_text("Không tìm thấy cột 'hour'.")

    # 2. MA TRẬN TƯƠNG QUAN
    if enable_streamlit: st.markdown("### 3. Ma trận tương quan")
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_plot = [c for c in num_cols if 'merchant_' not in c and 'country_' not in c and 'id' not in c]
    if 'is_fraud' not in cols_to_plot and 'is_fraud' in df.columns: cols_to_plot.append('is_fraud')
        
    corr = df[cols_to_plot].corr()

    fig4, ax4 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, ax=ax4)
    ax4.set_title('Correlation Matrix')
    show_fig(fig4)

    # 3. CHANNEL & MERCHANT
    if enable_streamlit: st.markdown("### 4. Phân tích Channel & Merchant")
    
    df_viz = df.copy()
    merch_cols = [c for c in df.columns if 'merchant_category_' in c]
    col_cat = None
    if merch_cols:
        df_viz['merchant_category_label'] = df_viz[merch_cols].idxmax(axis=1).apply(lambda x: x.replace('merchant_category_', ''))
        col_cat = 'merchant_category_label'
    elif 'merchant_category' in df.columns:
        col_cat = 'merchant_category'

    if col_cat:
        fraud_by_cat = df_viz.groupby(col_cat)['is_fraud'].mean().sort_values(ascending=False)
        fig5, ax5 = plt.subplots(figsize=(10, 5))
        fraud_by_cat.plot(kind='bar', color='orange', edgecolor='black', ax=ax5)
        ax5.set_title('Tỷ lệ Gian lận theo Merchant Category')
        plt.xticks(rotation=45)
        show_fig(fig5)

    if 'channel' in df_viz.columns:
        if pd.api.types.is_numeric_dtype(df_viz['channel']):
             df_viz['channel_label'] = df_viz['channel'].map({1: 'Web', 0: 'App'})
        else:
             df_viz['channel_label'] = df_viz['channel']
        
        fraud_by_channel = df_viz.groupby('channel_label')['is_fraud'].mean().sort_values(ascending=False)
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        fraud_by_channel.plot(kind='bar', color='purple', edgecolor='black', ax=ax6)
        ax6.set_title('Tỷ lệ Gian lận theo Channel')
        show_fig(fig6)

    # 4. SỐ TIỀN
    if enable_streamlit: st.markdown("### 5. Phân phối số tiền (Amount)")
    
    if 'amount' in df.columns:
        amount = df['amount'].astype(float)
        amount_pos = amount[amount > 0] 
        log_amount = np.log(amount_pos)

        fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax7a.hist(amount_pos, bins=60, color='green', alpha=0.7)
        ax7a.set_title('Phân phối số tiền (Gốc)')
        
        ax7b.hist(log_amount, bins=60, color='teal', alpha=0.7)
        ax7b.set_title('Phân phối số tiền (Log Transform)')
        show_fig(fig7)

# =============================================================================
# MODULE: HIDDEN PATTERNS
# =============================================================================

def analyze_hidden_patterns(model, X_test, y_test, feature_names, enable_streamlit=False):
    
    def show_fig(fig):
        if enable_streamlit: 
            st.pyplot(fig)
            plt.close(fig)
        else: pass

    if enable_streamlit: st.markdown("### 6. Phân tích Hidden Patterns (Top Features)")
    else: print(f"\n{'='*20} [EDA] PHÂN TÍCH HIDDEN PATTERNS {'='*20}")
    
    try:
        importances = model.feature_importances_
    except AttributeError:
        if enable_streamlit: st.warning("Model này không hỗ trợ Feature Importance.")
        return

    indices = np.argsort(importances)[::-1]
    top_n = min(4, len(feature_names))
    top_features = feature_names[indices][:top_n]
    
    df_plot = pd.DataFrame(X_test, columns=feature_names).copy()
    df_plot['is_fraud'] = y_test.values

    fig8, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.suptitle('PHÂN TÍCH ĐẶC ĐIỂM GIAO DỊCH GIAN LẬN', fontsize=16)
    axes = axes.flatten()

    for i, col in enumerate(top_features):
        if i >= len(axes): break
        sns.kdeplot(data=df_plot[df_plot['is_fraud']==0], x=col, label='Normal', ax=axes[i], fill=True, color='blue', alpha=0.3)
        sns.kdeplot(data=df_plot[df_plot['is_fraud']==1], x=col, label='Fraud', ax=axes[i], fill=True, color='red', alpha=0.3)
        axes[i].set_title(f'Phân phối: {col}')
        axes[i].legend()
    
    plt.tight_layout()
    show_fig(fig8)