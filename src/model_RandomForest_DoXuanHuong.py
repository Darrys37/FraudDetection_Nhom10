from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score

def train_random_forest(X_train, y_train):
    print(f"\n{'='*20} [TRAINING] RANDOM FOREST {'='*20}")

    # --- BƯỚC 1: TRƯỚC KHI TUNING ---
    print("1. [BEFORE] Chạy Model Mặc định...")
    rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_base.fit(X_train, y_train)
    
    y_pred_base = rf_base.predict(X_train)
    base_recall = recall_score(y_train, y_pred_base)
    print(f"   -> Recall (Mặc định): {base_recall:.2%}")

    # --- BƯỚC 2: SAU KHI TUNING ---
    print("\n2. [AFTER] Đang chạy Tuning (Grid Search)...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'criterion': ['gini', 'entropy']
    }

    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=3,
        scoring='recall',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    # --- BƯỚC 3: SO SÁNH ---
    best_model = grid_search.best_estimator_
    y_pred_tuned = best_model.predict(X_train)
    tuned_recall = recall_score(y_train, y_pred_tuned)

    print(f"   -> Tham số tốt nhất: {grid_search.best_params_}")
    print(f"   -> Recall (Sau Tuning): {tuned_recall:.2%}")
    print(f"==> HIỆU QUẢ TUNING: {(tuned_recall - base_recall):+.2%}")

    return best_model