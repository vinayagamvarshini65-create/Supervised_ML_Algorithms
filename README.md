---

# Email Spam Classification using Supervised Machine Learning

## 📌 Project Overview

This project focuses on the development of a robust **End-to-End Machine Learning Pipeline** to classify emails as **Spam** or **Ham** (Legitimate). By leveraging Natural Language Processing (NLP) and a suite of Supervised Learning algorithms, the system achieves high accuracy in identifying unsolicited communications.

The final product includes a **Streamlit Web Application** that allows users to input email text and receive real-time predictions.

## 🌐 Live Demo
You can test the trained model instantly without installing any local dependencies. 

**Deployment Link:** [👉 Click here to open the Web App](https://supervisedmlalgorithms-2cci4hjjqbrekfrhkgm2jh.streamlit.app/)

*Note: The app may take a few seconds to "wake up" if it hasn't been used recently.*

## 🛠️ Technical Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Joblib
* **NLP:** TF-IDF Vectorization
* **Frontend:** Streamlit
* **Algorithms Evaluated:** * Linear/Logistic Regression
* Naive Bayes (Multinomial)
* k-Nearest Neighbors (k-NN)
* Decision Tree
* Random Forest (Final Model)
* Ensemble Methods: Bagging & AdaBoost



## 📊 Methodology

### 1. Data Preprocessing & EDA

* **Dataset:** `spam_ham_dataset.csv` containing over 5,000 labeled emails.
* **Cleaning:** Removal of redundant columns and handling missing values.
* **Visualization:** Analysis of class distribution and word frequency.

### 2. Feature Engineering

Since machine learning models cannot process raw text, we utilized **TF-IDF (Term Frequency-Inverse Document Frequency)**. This technique converts text into a numerical matrix, highlighting words that are uniquely characteristic of spam versus legitimate mail.

### 3. Model Training & Comparison

We implemented a "Battle of Algorithms" approach, training each model individually to compare performance metrics. **Random Forest** was selected as the production model due to its superior F1-score and ability to handle high-dimensional text data without overfitting.

### 4. Deployment

The trained model and vectorizer were serialized using `joblib` and integrated into a Streamlit dashboard.

## 🚀 Installation & Usage

### Prerequisites

Clone this repository and ensure you have Python installed.

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt

```

### Step 2: Train the Model

Run the Jupyter Notebook or the Python training script to generate the serialized files:

* `spam_model.pkl`
* `vectorizer.pkl`

### Step 3: Launch the App

```bash
streamlit run app.py

```

## 📈 Performance Results

| Algorithm | Accuracy |
| --- | --- |
| **Random Forest** | **98.2%** |
| Logistic Regression | 97.5% |
| Naive Bayes | 94.1% |
| AdaBoost | 96.8% |

## 📂 Project Structure

```text
├── data/
│   └── spam_ham_dataset.csv    # Raw dataset
├── models/
│   ├── spam_model.pkl          # Trained Random Forest model
│   └── vectorizer.pkl          # TF-IDF Vectorizer
├── app.py                      # Streamlit application code
├── notebook.ipynb              # EDA and Model Training
└── requirements.txt            # Project dependencies

```
