import streamlit as st
import pandas as pd
import re
from urllib.parse import urlparse
import joblib

# Load trained model
clf = joblib.load("url_detector_model.pkl")

# Feature extraction
def extract_features(url):
    features = {}
    parsed = urlparse(url)
    
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_at'] = url.count('@')
    features['has_ip'] = int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', url)))
    features['https'] = int(parsed.scheme == 'https')
    
    domain_parts = parsed.netloc.split('.')
    features['subdomain_count'] = len(domain_parts) - 2 if len(domain_parts) > 2 else 0
    
    suspicious_keywords = ['login','secure','update','verify','account']
    features['suspicious_words'] = sum([1 for word in suspicious_keywords if word in url.lower()])
    
    return pd.Series(features)

# Prediction
def predict_url(url):
    features = extract_features(url).values.reshape(1, -1)
    return clf.predict(features)[0]

# Streamlit UI
st.title("URL Detector")
st.write("Enter a URL to check if it is benign, phishing, or malicious:")

url_input = st.text_input("Enter URL here")

if st.button("Predict"):
    if url_input:
        result = predict_url(url_input)
        st.success(f"Prediction: {result}")
    else:
        st.error("Please enter a URL to predict!")
