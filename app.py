import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Fraud Detection Task 2", layout="wide")
st.title("Task 2: Credit Card Fraud Detection (Imbalance Handling)")

@st.cache_data
def generate_data():
    np.random.seed(42)
    n_samples = 10000
    
    amount = np.random.exponential(scale=100, size=n_samples)
    v1 = np.random.normal(0, 1, n_samples)
    v2 = np.random.normal(0, 1, n_samples)
    v3 = np.random.normal(0, 1, n_samples)
    
    df = pd.DataFrame({
        "Amount": amount,
        "V1": v1,
        "V2": v2,
        "V3": v3
    })
    
    df['Class'] = 0
    fraud_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    
    df.loc[fraud_indices, 'Class'] = 1
    df.loc[fraud_indices, 'V1'] += 2.5
    df.loc[fraud_indices, 'V2'] -= 2.5
    df.loc[fraud_indices, 'Amount'] += 200
    
    return df

df = generate_data()

st.header("1. Exploratory Data Analysis")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Class Imbalance")
    class_counts = df['Class'].value_counts().reset_index()
    class_counts.columns = ['Class', 'Count']
    fig_pie = px.pie(class_counts, values='Count', names='Class', title='Fraud vs Non-Fraud Distribution', color='Class', color_discrete_map={0:'blue', 1:'red'})
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("Feature Distribution")
    feature = st.selectbox("Select Feature to Visualize", ["Amount", "V1", "V2", "V3"])
    fig_hist = px.histogram(df, x=feature, color="Class", barmode="overlay", title=f"{feature} Distribution by Class")
    st.plotly_chart(fig_hist, use_container_width=True)

st.header("2. Model Development & Imbalance Handling")

X = df.drop("Class", axis=1)
y = df['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model_choice = st.radio("Choose Model", ["Logistic Regression (Baseline)", "Random Forest (Ensemble)"])
handle_imbalance = st.checkbox("Handle Class Imbalance (Use Class Weights)", value=True)

class_weight_setting = 'balanced' if handle_imbalance else None

if model_choice == "Logistic Regression (Baseline)":
    model = LogisticRegression(class_weight=class_weight_setting)
else:
    model = RandomForestClassifier(n_estimators=100, class_weight=class_weight_setting, random_state=42)

model.fit(X_train, y_train)

st.header("3. Evaluation & Analysis")

threshold = st.slider("Classification Threshold (Optimization)", 0.0, 1.0, 0.5)

y_probs = model.predict_proba(X_test)[:, 1]
y_pred = (y_probs >= threshold).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{acc:.2%}")
m2.metric("Precision", f"{prec:.2%}")
m3.metric("Recall", f"{rec:.2%}")
m4.metric("F1 Score", f"{f1:.2%}")

c1, c2 = st.columns(2)

with c1:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(pd.DataFrame(cm, columns=["Predicted Safe", "Predicted Fraud"], index=["Actual Safe", "Actual Fraud"]))

with c2:
    st.subheader("Critical Thinking: Business Impact")
    st.info(f"""
    **Current Threshold: {threshold}**
    * **False Negatives (Missed Fraud):** {cm[1][0]} transactions. (High Risk)
    * **False Positives (Annoyed Customers):** {cm[0][1]} transactions. (Customer Friction)
    * **Recommendation:** Lower the threshold to catch more fraud if the cost of missed fraud is high.
    """)

if model_choice == "Random Forest (Ensemble)":
    st.subheader("Feature Importance")
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    st.bar_chart(importances.set_index('Feature'))