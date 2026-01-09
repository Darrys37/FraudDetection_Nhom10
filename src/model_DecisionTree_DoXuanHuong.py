from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def train_decision_tree(X_train, y_train):
    """Huấn luyện Decision Tree với GridSearch"""
    print(f"\n{'='*20} [TRAINING] DECISION TREE {'='*20}")

    # 1. Thiết lập tham số
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }

    dt_base = DecisionTreeClassifier(random_state=42, class_weight='balanced')

    # 2. Chạy Grid Search
    print("-> Đang chạy Grid Search cho Decision Tree...")
    grid_search = GridSearchCV(
        estimator=dt_base,
        param_grid=param_grid,
        cv=3,
        scoring='recall',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(f"-> Tham số tốt nhất DT: {grid_search.best_params_}")
    print(f"-> Recall tốt nhất (Train): {grid_search.best_score_:.2%}")

    return grid_search.best_estimator_