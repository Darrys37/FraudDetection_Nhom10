import streamlit as st
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # <--- Import thêm joblib
from sklearn.metrics import recall_score, f1_score, accuracy_score, confusion_matrix, classification_report

# Thêm đường dẫn src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.preprocessing import load_and_clean_data, prepare_data_for_model
from src.feature_engineering import create_new_features
from src.model_RandomForest_DoXuanHuong import train_random_forest
from src.model_XGBoost_NguyenHuynhAnhTuan import train_xgboost
from src.model_DecisionTree_TranTatPhat import train_decision_tree
from src.evaluation import evaluate_model, plot_feature_importance
from src.eda import run_eda_analysis, analyze_hidden_patterns

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# Hàm hỗ trợ Load hoặc Train Model
def get_model(model_name, X_train, y_train):
    """
    Cố gắng load model từ file .pkl trước.
    Nếu không tìm thấy file, mới bắt buộc phải train lại.
    """
    file_path = ""
    if model_name == "Decision Tree":
        file_path = "models/decision_tree.pkl"
    elif model_name == "Random Forest":
        file_path = "models/random_forest.pkl"
    elif model_name == "XGBoost":
        file_path = "models/xgboost.pkl"
    
    # 1. Thử Load file
    if os.path.exists(file_path):
        # st.success(f"⚡ Đã load model {model_name} từ file (Không cần train lại!)")
        return joblib.load(file_path)
    
    # 2. Nếu không có file thì Train mới
    st.warning(f"Không thấy file {file_path}. Đang huấn luyện lại từ đầu...")
    model = None
    if model_name == "Decision Tree":
        model = train_decision_tree(X_train, y_train)
    elif model_name == "Random Forest":
        model = train_random_forest(X_train, y_train)
    elif model_name == "XGBoost":
        model = train_xgboost(X_train, y_train)
    
    # Tiện tay lưu luôn để lần sau đỡ cực
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(model, file_path)
    
    return model

def display_model_details(model, model_name, X_test, y_test, feature_names):
    st.markdown(f"### Chi tiết đánh giá: {model_name}")
    
    fig_cm, fig_roc = evaluate_model(model, X_test, y_test, model_name)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Confusion Matrix**")
        st.pyplot(fig_cm)
    with col_b:
        st.write("**ROC Curve**")
        if fig_roc: st.pyplot(fig_roc)
        else: st.warning("Model này không hỗ trợ vẽ ROC.")

    st.markdown("---")
    st.write("**Top 10 Đặc trưng quan trọng (Feature Importance)**")
    fig_feat = plot_feature_importance(model, feature_names, model_name)
    if fig_feat: st.pyplot(fig_feat)
    else: st.warning("Model này không hỗ trợ Feature Importance.")

    st.markdown("---")
    analyze_hidden_patterns(model, X_test, y_test, feature_names, enable_streamlit=True)


def main():
    st.title("DASHBOARD PHÁT HIỆN GIAN LẬN GIAO DỊCH (NHÓM 10)")
    st.markdown("---")

    st.sidebar.header("Bảng điều khiển")
    
    model_choice = st.sidebar.selectbox(
        "Chọn Chế độ chạy:",
        ("So sánh tất cả Model", "Random Forest", "XGBoost", "Decision Tree")
    )

    uploaded_file = st.sidebar.file_uploader("Upload file (.csv)", type=["csv"])

    if uploaded_file:
        st.success("File đã được tải!")
        df_origin = pd.read_csv(uploaded_file)
        
        with st.expander("Xem dữ liệu gốc"):
            st.dataframe(df_origin.head())

        if st.sidebar.button("CHẠY PHÂN TÍCH & DỰ ĐOÁN NGAY"):
            
            # 1. EDA
            st.header("1. PHÂN TÍCH DỮ LIỆU KHÁM PHÁ (EDA)")
            with st.spinner('Đang xử lý dữ liệu & Vẽ biểu đồ...'):
                temp_path = "temp_uploaded.csv"
                df_origin.to_csv(temp_path, index=False)
                df_clean = load_and_clean_data(temp_path)
                df_features = create_new_features(df_clean)
                run_eda_analysis(df_features, enable_streamlit=True)
                X_train, X_test, y_train, y_test, feature_names = prepare_data_for_model(df_features)
            
            st.markdown("---")
            st.header("2. KẾT QUẢ HUẤN LUYỆN MÔ HÌNH")

            # --- TRƯỜNG HỢP 1: SO SÁNH TẤT CẢ ---
            if model_choice == "So sánh tất cả Model":
                st.info("Đang tải các mô hình để so sánh...")
                
                results = []
                models_dict = {}

                # Load 3 model (Cực nhanh nếu đã có file)
                dt = get_model("Decision Tree", X_train, y_train)
                y_pred_dt = dt.predict(X_test)
                results.append({"Model": "Decision Tree", "Recall": recall_score(y_test, y_pred_dt), "F1": f1_score(y_test, y_pred_dt)})
                models_dict["Decision Tree"] = dt

                rf = get_model("Random Forest", X_train, y_train)
                y_pred_rf = rf.predict(X_test)
                results.append({"Model": "Random Forest", "Recall": recall_score(y_test, y_pred_rf), "F1": f1_score(y_test, y_pred_rf)})
                models_dict["Random Forest"] = rf

                xgb = get_model("XGBoost", X_train, y_train)
                y_pred_xgb = xgb.predict(X_test)
                results.append({"Model": "XGBoost", "Recall": recall_score(y_test, y_pred_xgb), "F1": f1_score(y_test, y_pred_xgb)})
                models_dict["XGBoost"] = xgb

                # Hiện kết quả
                st.subheader("Bảng Xếp Hạng Hiệu Quả")
                df_res = pd.DataFrame(results).set_index("Model")
                st.dataframe(df_res.style.highlight_max(axis=0, color='lightgreen'))
                st.bar_chart(df_res['Recall'])

                st.markdown("---")
                st.subheader("3. CHI TIẾT TỪNG MÔ HÌNH")
                
                tab1, tab2, tab3 = st.tabs(["Decision Tree", "Random Forest", "XGBoost"])
                with tab1: display_model_details(models_dict["Decision Tree"], "Decision Tree", X_test, y_test, feature_names)
                with tab2: display_model_details(models_dict["Random Forest"], "Random Forest", X_test, y_test, feature_names)
                with tab3: display_model_details(models_dict["XGBoost"], "XGBoost", X_test, y_test, feature_names)

            # --- TRƯỜNG HỢP 2: CHẠY 1 MODEL LẺ ---
            else:
                with st.spinner(f'Đang tải {model_choice}...'):
                    model = get_model(model_choice, X_train, y_train)
                    
                    y_pred = model.predict(X_test)
                    col1, col2 = st.columns(2)
                    col1.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")
                    col2.metric("F1-Score", f"{f1_score(y_test, y_pred):.2%}")

                    st.markdown("---")
                    display_model_details(model, model_choice, X_test, y_test, feature_names)

    else:
        st.info("Vui lòng tải file CSV để bắt đầu.")

if __name__ == "__main__":
    main()