import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# Streamlit Page Config
st.set_page_config(layout="wide", page_title="Universal Bank - Personal Loan Dashboard")
st.title("💰 Universal Bank: Personal Loan Prediction Dashboard")

# Sidebar Navigation
tab = st.sidebar.radio("Select Section", [
    "📊 Marketing Insights",
    "🤖 Model Evaluation",
    "📂 Predict New Data"
])

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("universal_bank.csv")

data = load_data()

# ------------------------------------------------------
# 📊 1. Marketing Insights
# ------------------------------------------------------
if tab == "📊 Marketing Insights":
    st.header("Customer Insights Dashboard")
    st.write("Explore patterns to improve personal loan conversions.")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.countplot(x='Personal Loan', data=data, palette='coolwarm', ax=ax)
        ax.set_title("Personal Loan Acceptance Count")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(x='Personal Loan', y='Income', data=data, palette='viridis', ax=ax)
        ax.set_title("Income vs Personal Loan")
        st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.barplot(x='Education', y='Personal Loan', data=data, palette='plasma', estimator=np.mean)
    ax.set_title("Education Level vs Loan Acceptance Rate")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.histplot(data['CCAvg'], kde=True, color='teal')
    ax.set_title("Distribution of Credit Card Spending (CCAvg)")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    corr = data.corr(numeric_only=True)
    sns.heatmap(corr, cmap='YlGnBu', annot=True, fmt=".2f")
    ax.set_title("Correlation Heatmap of Features")
    st.pyplot(fig)

# ------------------------------------------------------
# 🤖 2. Model Evaluation
# ------------------------------------------------------
elif tab == "🤖 Model Evaluation":
    st.header("Model Comparison and Performance Metrics")
    st.write("Click below to train all models and compare their performance.")

    if st.button("Run Models and Show Results"):
        X = data.drop(['ID', 'Personal Loan'], axis=1)
        y = data['Personal Loan']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosted Tree": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        }

        results = []
        fig_roc, ax_roc = plt.subplots()

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            y_proba = model.predict_proba(X_test)[:, 1]

            # Metrics
            results.append({
                "Algorithm": name,
                "Train Accuracy": accuracy_score(y_train, y_pred_train),
                "Test Accuracy": accuracy_score(y_test, y_pred_test),
                "Precision": precision_score(y_test, y_pred_test),
                "Recall": recall_score(y_test, y_pred_test),
                "F1-Score": f1_score(y_test, y_pred_test),
                "AUC": roc_auc_score(y_test, y_proba)
            })

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_proba):.2f})")

            # Confusion Matrices
            for dataset, pred, title in [
                ("Train", y_pred_train, f"{name} - Training Confusion Matrix"),
                ("Test", y_pred_test, f"{name} - Testing Confusion Matrix")
            ]:
                cm = confusion_matrix(y_train if dataset=="Train" else y_test,
                                      pred)
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                ax_cm.set_title(title)
                st.pyplot(fig_cm)

            # Feature Importance
            if hasattr(model, "feature_importances_"):
                importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                fig_imp, ax_imp = plt.subplots()
                importance.plot(kind='bar', ax=ax_imp, color='tomato')
                ax_imp.set_title(f"{name} - Feature Importance")
                st.pyplot(fig_imp)

        # Metrics Table
        st.subheader("📈 Model Performance Summary")
        st.dataframe(pd.DataFrame(results).round(3))

        # ROC Curve Combined
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve Comparison")
        ax_roc.legend()
        st.pyplot(fig_roc)

# ------------------------------------------------------
# 📂 3. Predict New Data
# ------------------------------------------------------
elif tab == "📂 Predict New Data":
    st.header("Upload New Data to Predict Personal Loan")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        st.subheader("Preview of Uploaded Data")
        st.dataframe(new_data.head())

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(data.drop(['ID','Personal Loan'], axis=1), data['Personal Loan'])
        preds = model.predict(new_data.drop(['ID'], axis=1, errors='ignore'))
        new_data['Personal Loan Prediction'] = preds

        st.subheader("Predicted Results")
        st.dataframe(new_data.head())

        csv = new_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Predictions as CSV",
            data=csv,
            file_name='predicted_personal_loans.csv',
            mime='text/csv'
        )
