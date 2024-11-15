import streamlit as st 
import pandas as pd 
import os 
import joblib
from pathlib import Path 


# set the page title 
st.set_page_config(page_title="Sentiment Analysis ",page_icon=":shamrock:")

# app title and description 
st.title("Sentiment Analysis ")
st.write("This app predicts the sentiment of text based on different sources.")

# define the path to the models directory using pathlib for cross.platform compatibility 
models_dir = Path('src/models')

# Check if the models directory exists
if not models_dir.exists():
    st.error(f"Models directory not found at {models_dir.resolve()}. Please ensure the path is correct.")
    st.stop()
    
    
# Load available sources
try:
    sources = [file.stem.split('_')[0] for file in models_dir.glob('*_model.joblib')]
    if not sources:
        st.error("No models found in the models directory. Please add model files with the naming convention '<source>_model.joblib'.")
        st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()
    
# User input
selected_source = st.selectbox("Select a source:", sources)
user_text = st.text_area("Enter your text here:")


def load_model_and_vectorizer(source):
    try:
        # Load the trained model
        model_path = f'src/models/{source}_model.joblib'
        vectorizer_path = f'src/models/{source}_vectorizer.joblib'
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)  # Load the vectorizer used for training
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        return None, None

def predict_sentiment(text, source):
    model, vectorizer = load_model_and_vectorizer(source)
    if model is None or vectorizer is None:
        return "Error"
    try:
        # Transform the input text to the numeric format using the vectorizer
        transformed_text = vectorizer.transform([text])
        
        # Make the prediction using the transformed text
        prediction = model.predict(transformed_text)[0]
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Error"
    
    
    
if st.button("Predict Sentiment"):
    if user_text.strip():
        # Perform sentiment prediction
        sentiment = predict_sentiment(user_text, selected_source)
        
        # Display result
        st.subheader("Prediction Result:")
        if sentiment == "Positive":
            st.success(f"The sentiment is {sentiment} 😊")
        elif sentiment == "Negative":
            st.error(f"The sentiment is {sentiment} 😞")
        elif sentiment == "Neutral":
            st.info(f"The sentiment is {sentiment} 😐")
        else:
            st.warning(f"Received unexpected sentiment: {sentiment}")
    else:
        st.warning("Please enter some text for prediction.")
        
        
# Display some sample texts
st.subheader("Sample Texts:")
sample_data_path = Path("data/twitter_validation.csv")

# Check if the sample data file exists
if not sample_data_path.exists():
    st.error(f"Sample data file not found at {sample_data_path.resolve()}. Please ensure the file exists.")
else:
    try:
        samples = pd.read_csv(sample_data_path, header=None, names=["serial_number", "Source", "Sentiment", "Text"])
        if samples.empty:
            st.warning("Sample data file is empty.")
        else:
            st.dataframe(samples[['Source', 'Sentiment', 'Text']].sample(min(10, len(samples))))
    except Exception as e:
        st.error(f"Error loading sample data: {e}")

