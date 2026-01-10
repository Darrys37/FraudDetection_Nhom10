from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score

def train_decision_tree(X_train, y_train):
    print(f"\n{'='*20} [TRAINING] DECISION TREE {'='*20}")

    # --- BƯỚC 1: TRƯỚC KHI TUNING (Model Mặc định) ---
    print("1. [BEFORE] Chạy Model Mặc định (Default Params)...")
    dt_base = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    dt_base.fit(X_train, y_train)
    
    y_pred_base = dt_base.predict(X_train)
    base_recall = recall_score(y_train, y_pred_base)
    print(f"   -> Recall (Mặc định): {base_recall:.2%}")

    # --- BƯỚC 2: SAU KHI TUNING (Grid Search) ---
    print("\n2. [AFTER] Đang chạy Tuning (Grid Search)...")
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(
        estimator=dt_base,
        param_grid=param_grid,
        cv=3,
        scoring='recall',
        n_jobs=-1
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