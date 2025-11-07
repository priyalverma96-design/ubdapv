# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

st.set_page_config(layout="wide", page_title="Universal Bank Personal Loan Dashboard")

st.title("Universal Bank - Personal Loan Analytics & Prediction Dashboard")

# Sidebar tabs
tab = st.sidebar.radio("Select Page", ["Marketing Insights", "Model Evaluation", "Predict New Data"])

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("universal_bank.csv")  # replace with your dataset
    return data

data = load_data()

# ----------------- Marketing Insights ----------------- #
if tab == "Marketing Insights":
    st.header("Customer Insights for Marketing Decisions")
    
    st.markdown("### Dataset Overview")
    st.dataframe(data.head())
    
    st.markdown("### Customer Distribution by Personal Loan")
    fig, ax = plt.subplots()
    sns.countplot(x='Personal Loan', data=data, palette='coolwarm', ax=ax)
    ax.set_title("Personal Loan Acceptance Count")
    st.pyplot(fig)
    
    st.markdown("### Income vs Personal Loan Acceptance")
    fig, ax = plt.subplots()
    sns.boxplot(x='Personal Loan', y='Income', data=data, palette='Set2', ax=ax)
    ax.set_title("Income Distribution by Personal Loan Acceptance")
    st.pyplot(fig)
    
    st.markdown("### Credit Card Spending vs Personal Loan Acceptance")
    fig, ax = plt.subplots()
    sns.boxplot(x='Personal Loan', y='CCAvg', data=data, palette='Set1', ax=ax)
    ax.set_title("Average Credit Card Spending by Loan Acceptance")
    st.pyplot(fig)
    
    st.markdown("### Education Level vs Loan Acceptance")
    fig, ax = plt.subplots()
    sns.countplot(x='Education', hue='Personal Loan', data=data, palette='Set3', ax=ax)
    ax.set_title("Loan Acceptance by Education Level")
    st.pyplot(fig)
    
    st.markdown("### Family Size vs Personal Loan Acceptance")
    fig, ax = plt.subplots()
    sns.countplot(x='Family', hue='Personal Loan', data=data, palette='pastel', ax=ax)
    ax.set_title("Loan Acceptance by Family Size")
    st.pyplot(fig)

# ----------------- Model Evaluation ----------------- #
elif tab == "Model Evaluation":
    st.header("Train and Evaluate Models")
    
    if st.button("Run Models and Generate Metrics"):
        st.write("Running Decision Tree, Random Forest, Gradient Boosted Tree...")
        
        # Prepare data
        X = data.drop(['ID','Personal Loan'], axis=1)
        y = data['Personal Loan']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosted Tree': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        }
        
        results = pd.DataFrame(columns=['Algorithm', 'Train Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'])
        plt.figure(figsize=(8,6))
        colors = ['blue','green','red']
        
        for i,(name, model) in enumerate(models.items()):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
            
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            y_test_proba = model.predict_proba(X_test)[:,1]
            
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred)
            recall = recall_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred)
            auc = roc_auc_score(y_test, y_test_proba)
            
            results = pd.concat([results, pd.DataFrame([[name, train_acc, test_acc, precision, recall, f1, auc]], columns=results.columns)], ignore_index=True)
            
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            plt.plot(fpr, tpr, color=colors[i], label=f'{name} (AUC={auc:.2f})')
            
            # Confusion matrices
            fig, axes = plt.subplots(1,2, figsize=(12,5))
            sns.heatmap(confusion_matrix(y_train, y_train_pred), annot=True, fmt='d', ax=axes[0], cmap='Blues')
            axes[0].set_title(f'{name} - Training Confusion Matrix')
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('Actual')
            
            sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', ax=axes[1], cmap='Greens')
            axes[1].set_title(f'{name} - Testing Confusion Matrix')
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('Actual')
            st.pyplot(fig)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feat_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(8,6))
                sns.barplot(x=feat_importance.values, y=feat_importance.index, palette='viridis', ax=ax)
                ax.set_title(f'{name} - Feature Importance')
                st.pyplot(fig)
        
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - All Models')
        plt.legend()
        st.pyplot(plt)
        
        st.markdown("### Metrics Table")
        st.dataframe(results)

# ----------------- Predict New Data ----------------- #
elif tab == "Predict New Data":
    st.header("Upload Dataset and Predict Personal Loan")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset Preview:")
        st.dataframe(new_data.head())
        
        X_new = new_data.drop(['ID'], axis=1, errors='ignore')
        
        # Use trained Random Forest as default predictor
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(data.drop(['ID','Personal Loan'], axis=1), data['Personal Loan'])
        
        new_data['Personal Loan Prediction'] = model.predict(X_new)
        st.write("Predictions:")
        st.dataframe(new_data.head())
        
        csv = new_data.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Predictions", data=csv, file_name='predicted_personal_loans.csv', mime='text/csv')
