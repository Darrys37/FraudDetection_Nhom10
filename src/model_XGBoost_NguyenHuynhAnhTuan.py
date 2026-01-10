from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import warnings

# Tắt cảnh báo XGBoost spam log
warnings.filterwarnings('ignore', category=UserWarning)

def train_xgboost(X_train, y_train):
    """
    Huấn luyện XGBoost với GridSearchCV (Quét toàn bộ tham số).
    Code này chạy lâu hơn nhưng tìm ra tham số tối ưu hơn RandomizedSearch.
    """
    print(f"\n{'='*20} [TRAINING] XGBOOST (GRID SEARCH) {'='*20}")

    # 1. Tính scale_pos_weight (Xử lý mất cân bằng dữ liệu)
    # Vì chúng ta đã undersample ở preprocessing nên tỷ lệ là 1:10 (hoặc xấp xỉ)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    xgb_base = XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        use_label_encoder=False
    )

    # 2. Thiết lập không gian tham số (Grid Search - Theo yêu cầu mới của bạn)
    # Đây là bộ tham số bạn vừa cung cấp trong đoạn code văn bản
    param_grid = {
        'n_estimators': [100, 200, 300],      
        'max_depth': [6, 8, 10],         
        'learning_rate': [0.05, 0.1, 0.2], 
        'subsample': [0.8, 1.0],         
        'colsample_bytree': [0.8, 1.0]   
    }

    # 3. Chạy Grid Search
    print("-> Đang chạy Grid Search cho XGBoost (sẽ tốn thời gian, vui lòng đợi)...")
    
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='recall', # Ưu tiên bắt được gian lận
        cv=5,             # Tăng lên 5 fold như code bạn yêu cầu
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(f"-> Tham số tốt nhất XGB: {grid_search.best_params_}")
    print(f"-> Recall tốt nhất (Train): {grid_search.best_score_:.2%}")

    return grid_search.best_estimator_