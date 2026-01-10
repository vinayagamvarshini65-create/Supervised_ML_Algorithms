import streamlit as st
import joblib

# Set Page Config
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧")

# 1. LOAD ASSETS
# Use st.cache_resource to load the model once and keep it in memory
@st.cache_resource
def load_assets():
    print("Log: Attempting to load 'spam_model.pkl' and 'vectorizer.pkl'...")
    try:
        model = joblib.load('spam_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        print("Log: Model and Vectorizer loaded successfully.")
        return model, vectorizer
    except Exception as e:
        print(f"Log Error: Could not load files. {e}")
        return None, None

model, vectorizer = load_assets()

# 2. USER INTERFACE
st.title("📧 Email Spam Classifier")
st.write("This application uses Machine Learning to detect if an email is **Spam** or **Ham**.")

# Text input area
user_input = st.text_area("Paste the email text here:", height=200)

# 3. PREDICTION LOGIC
if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text to classify.")
        print("Log: Prediction attempt with empty input.")
    elif model is None or vectorizer is None:
        st.error("Model files not found. Please ensure .pkl files are in the directory.")
    else:
        print(f"Log: Processing input text (Length: {len(user_input)})")
        
        # Transform the input text using the loaded vectorizer
        # This converts text into the numerical format the model understands
        vectorized_text = vectorizer.transform([user_input])
        
        # Make the prediction
        prediction = model.predict(vectorized_text)[0]
        
        # Display the result
        if prediction == 1:
            st.error("### 🚨 Result: This email is SPAM")
            print("Log: Result -> SPAM")
        else:
            st.success("### ✅ Result: This email is HAM (Safe)")
            print("Log: Result -> HAM")

# Sidebar information
st.sidebar.subheader("How it works")
st.sidebar.write("1. The text is processed by a TF-IDF Vectorizer.")
st.sidebar.write("2. A Random Forest Classifier predicts the category.")
st.sidebar.write("3. The result is displayed as Spam or Ham.")

# 5. SIDEBAR INFO
st.sidebar.header("Project Details")
st.sidebar.info("""
**Algorithms used for comparison:**
- Logistic Regression
- Naive Bayes
- k-NN
- Decision Tree
- Random Forest (Selected)
- Bagging & AdaBoost
""")