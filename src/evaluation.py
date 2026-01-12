import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Trả về 2 biểu đồ: Confusion Matrix và ROC Curve để Streamlit vẽ
    """
    print(f"\n{'='*20} [EVALUATION] ĐÁNH GIÁ: {model_name.upper()} {'='*20}")

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = None

    # 1. In báo cáo ra Terminal (để debug)
    print(classification_report(y_test, y_pred))

    # 2. Vẽ Confusion Matrix
    fig_cm = plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}") 
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.close(fig_cm) # Đóng để không bị đè hình

    # 3. Vẽ ROC Curve
    fig_roc = None
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig_roc = plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.close(fig_roc)

    return fig_cm, fig_roc

def plot_feature_importance(model, feature_names, model_name="Model"):
    """
    Trả về biểu đồ Feature Importance để Streamlit vẽ
    """
    print(f"-> Đang vẽ Feature Importance cho {model_name}...")
    
    if not hasattr(model, 'feature_importances_'):
        print(f"Cảnh báo: {model_name} không hỗ trợ feature_importances_")
        return None

    importances = model.feature_importances_
    
    # Sắp xếp giảm dần và lấy top 10
    indices = np.argsort(importances)[::-1][:10]
    
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    # Vẽ biểu đồ
    fig_feat = plt.figure(figsize=(10, 5))
    palette = "viridis" if "Random" in model_name else "magma"
    
    sns.barplot(x=top_importances, y=top_features, palette=palette)
    plt.title(f"Top 10 Feature Importance - {model_name} (After Tuning)")
    plt.xlabel("Mức độ quan trọng")
    plt.tight_layout()
    plt.close(fig_feat)
    
    return fig_feat