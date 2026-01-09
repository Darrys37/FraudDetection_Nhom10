import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Đánh giá mô hình và vẽ biểu đồ (Không show ngay)"""
    print(f"\n{'='*20} [EVALUATION] ĐÁNH GIÁ: {model_name.upper()} {'='*20}")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 1. Classification Report
    print(classification_report(y_test, y_pred))

    # 2. Confusion Matrix
    plt.figure(figsize=(6, 5)) # Tạo cửa sổ mới
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}") 
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    # ĐÃ XÓA plt.show() Ở ĐÂY

    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5)) # Tạo cửa sổ mới
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    # ĐÃ XÓA plt.show() Ở ĐÂY