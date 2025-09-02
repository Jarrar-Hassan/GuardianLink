import pandas as pd
import re
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib  # to save the trained model

# Load dataset
data = pd.read_csv("urls.csv")  # CSV with 'url' and 'label'

# Feature extraction function
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

# Extract features
X = data['url'].apply(extract_features)
y = data['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, "url_detector_model.pkl")
print("Model saved as url_detector_model.pkl")
