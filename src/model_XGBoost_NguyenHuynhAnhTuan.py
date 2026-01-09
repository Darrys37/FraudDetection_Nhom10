from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import warnings

# Tắt cảnh báo XGBoost spam log
warnings.filterwarnings('ignore', category=UserWarning)

def train_xgboost(X_train, y_train):
    """Huấn luyện XGBoost với RandomizedSearchCV"""
    print(f"\n{'='*20} [TRAINING] XGBOOST {'='*20}")

    # 1. Tính scale_pos_weight
    # Vì chúng ta đã undersample ở preprocessing nên tỷ lệ là 1:10
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # [FIX] Bỏ 'use_label_encoder=False' vì phiên bản mới không cần và gây warning
    xgb_base = XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1
    )

    # 2. Thiết lập không gian tham số ngẫu nhiên
    param_dist = {
        'n_estimators': randint(100, 300),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.2),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3)
    }

    # 3. Chạy Randomized Search
    print("-> Đang chạy Randomized Search cho XGBoost...")
    
    xgb_random = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_dist,
        n_iter=20,  # Thử 20 tổ hợp ngẫu nhiên
        scoring='recall',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    xgb_random.fit(X_train, y_train)

    print(f"-> Tham số tốt nhất XGB: {xgb_random.best_params_}")
    print(f"-> Recall tốt nhất (Train): {xgb_random.best_score_:.2%}")

    return xgb_random.best_estimator_