# Task 2: Fraud Detection App 🛡️

This is my submission for the Google Developer Club recruitment task. It's a Machine Learning app that detects fraudulent transactions in real-time.

Instead of just checking for accuracy (which can be misleading since fraud is rare), this app focuses on **Precision and Recall** to actually catch bad transactions without blocking too many real ones.

## 💡 How it works
* **Auto-Generated Data:** I wrote a script to generate synthetic transaction data automatically using `numpy`. This means you don't need to download or upload any large CSV files to test it.
* **The Model:** I used a **Random Forest Classifier**. It's great for this kind of data because it handles complex patterns better than simple linear models.
* **Handling Imbalance:** Since fraud only happens ~2% of the time, I used "Class Weights" to tell the model that missing a fraud is a bigger mistake than flagging a safe one.

## 🚀 How to Run It
1.  **Install the libraries:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Start the app:**
    ```bash
    streamlit run app.py
    ```

## 🛠️ Tech Stack
* **Python** (Logic)
* **Streamlit** (User Interface)
* **Scikit-Learn** (ML Model)
* **Plotly** (Charts & Graphs)

## 🔍 Features to Check
* **Threshold Slider:** You can adjust the "strictness" of the model. Lowering it catches more fraud but might flag some safe payments (False Positives).
* **Confusion Matrix:** A visual way to see exactly how many frauds the model caught vs. missed.