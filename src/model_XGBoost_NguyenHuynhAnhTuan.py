from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import recall_score # <--- Thêm dòng này để tính điểm so sánh
import warnings

# Tắt cảnh báo XGBoost spam log
warnings.filterwarnings('ignore', category=UserWarning)

def train_xgboost(X_train, y_train):
    print(f"\n{'='*20} [TRAINING] XGBOOST (LOGIC GỐC + SO SÁNH) {'='*20}")

    # 1. Tính scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # 2. Khởi tạo Model Base (Giữ nguyên tham số gốc)
    xgb_base = XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        use_label_encoder=False
    )

    # --- [MỚI] BƯỚC SO SÁNH TRƯỚC KHI TUNING ---
    print("1. [BEFORE] Đang chạy Model Mặc định để so sánh...")
    xgb_base.fit(X_train, y_train)
    y_pred_base = xgb_base.predict(X_train)
    base_recall = recall_score(y_train, y_pred_base)
    print(f"   -> Recall (Mặc định): {base_recall:.2%}")
    # -------------------------------------------

    # 3. Thiết lập Param Grid (GIỮ NGUYÊN BẢN GỐC)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # 4. Chạy Grid Search (GIỮ NGUYÊN cv=5)
    print("\n2. [AFTER] Đang chạy Grid Search (cv=5)...")
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        cv=5,               # Giữ nguyên
        scoring='recall',   # Giữ nguyên
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    best_xgb = grid_search.best_estimator_
    
    # --- [MỚI] TÍNH TOÁN HIỆU QUẢ ---
    print("\nBest Parameters cho XGBoost:")
    print(grid_search.best_params_)
    
    y_pred_tuned = best_xgb.predict(X_train)
    tuned_recall = recall_score(y_train, y_pred_tuned)
    print(f"   -> Recall (Sau Tuning): {tuned_recall:.2%}")
    print(f"==> HIỆU QUẢ TUNING: {(tuned_recall - base_recall):+.2%}")
    # --------------------------------

    # 5. Cross-validation kiểm chứng (GIỮ NGUYÊN)
    print("\n-> Đang chạy Cross-validation (cv=10)...")
    cv_recall = cross_val_score(best_xgb, X_train, y_train, cv=10, scoring='recall')
    print(f"Mean Recall trên CV=10 (train): {cv_recall.mean():.2%} ± {cv_recall.std():.4f}")

    return best_xgb