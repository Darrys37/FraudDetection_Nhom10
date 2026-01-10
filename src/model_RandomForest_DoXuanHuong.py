from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score

def train_random_forest(X_train, y_train):
    """Huấn luyện Random Forest với GridSearch"""
    print(f"\n{'='*20} [TRAINING] RANDOM FOREST {'='*20}")

    # 1. Thiết lập Grid Search
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'criterion': ['gini', 'entropy']
    }

    rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)

    print("-> Đang chạy Grid Search (vui lòng đợi)...")
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=3, # Giảm xuống 3 fold cho nhanh
        scoring='recall',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    
    print(f"-> Tham số tốt nhất: {grid_search.best_params_}")
    print(f"-> Recall tốt nhất trên tập Train: {grid_search.best_score_:.2%}")

    return grid_search.best_estimator_