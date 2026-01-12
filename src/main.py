import sys
import os
import matplotlib.pyplot as plt 

# Đảm bảo Python nhìn thấy thư mục src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_and_clean_data, prepare_data_for_model
from src.feature_engineering import create_new_features
from src.model_DecisionTree_TranTatPhat import train_decision_tree  
from src.model_RandomForest_DoXuanHuong import train_random_forest 
from src.model_XGBoost_NguyenHuynhAnhTuan import train_xgboost
from src.evaluation import evaluate_model, plot_feature_importance
# --- SỬA DÒNG DƯỚI ĐÂY ---
from src.eda import analyze_hidden_patterns, run_eda_analysis 

def main():
    data_path = 'data/transactions.csv'
    
    # 1. Load & Process
    df = load_and_clean_data(data_path)
    df = create_new_features(df)
    
    # Chạy phân tích EDA tổng quan trước khi đưa vào model
    # Lưu ý: Khi biểu đồ hiện lên, mày phải tắt biểu đồ đi thì code mới chạy tiếp phần train model nhé.
    run_eda_analysis(df) 
    # ----------------------------------------

    X_train, X_test, y_train, y_test, feature_names = prepare_data_for_model(df)

    # --- ĐOẠN NÀY ĐỂ LƯU FILE ---
    output_path = 'data/transactions_processed_cleaned.csv'
    df.to_csv(output_path, index=False)
    print(f">>> Đã lưu file dữ liệu sạch tại: {output_path}")
    # ---------------------------------
    
    # 2. Train & Evaluate từng model (Code sẽ chạy một mạch, không bị dừng)
    
    # --- Decision Tree ---
    dt_model = train_decision_tree(X_train, y_train)
    evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    plot_feature_importance(dt_model, feature_names, "Decision Tree")

    # --- Random Forest ---
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, "Random Forest")
    plot_feature_importance(rf_model, feature_names, "Random Forest")

    # --- XGBoost ---
    xgb_model = train_xgboost(X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    plot_feature_importance(xgb_model, feature_names, "XGBoost")

    # 3. Analyze Hidden Patterns (Dùng Random Forest vì F1-score cao nhất)
    analyze_hidden_patterns(rf_model, X_test, y_test, feature_names)

    # 4. HIỂN THỊ TẤT CẢ BIỂU ĐỒ CÙNG LÚC
    print("\n-> Đang hiển thị tất cả biểu đồ (Dashboard)...")
    plt.show() # <--- QUAN TRỌNG: Gọi lệnh này ở cuối cùng

if __name__ == "__main__":
    main()