Of course. Here is the complete README file with the developer credit included at the end.

-----

# GuardianLink: AI-Powered Phishing & Malicious URL Detector

This project is a URL classifier that uses a machine learning model to detect and classify URLs as safe, malicious, or phishing attempts. It features a simple, interactive web interface built with Streamlit for real-time analysis and prediction.

## 🚀 Features

  * **Multi-Class Classification**: Classifies URLs into three distinct categories: `safe`, `malicious`, and `phishing`.
  * **Machine Learning Model**: Utilizes a `RandomForestClassifier` from `scikit-learn` to identify patterns in URL structures.
  * **Interactive Web UI**: A simple and user-friendly web application built with Streamlit allows for easy, real-time URL checks.
  * **Complete Workflow**: The repository includes the dataset, model training script, and the pre-trained model file.

## ⚙️ How It Works

The detection process is based on a machine learning pipeline that analyzes the lexical features of a URL.

1.  **Feature Extraction**: Each URL is converted into a set of numerical features that the model can understand. The features include:
      * Length of the URL
      * Number of dots (`.`), hyphens (`-`), and at-symbols (`@`)
      * Presence of an IP address in the domain
      * Number of subdomains
      * Presence of suspicious keywords (e.g., 'login', 'secure', 'verify', 'account', 'update')
2.  **Model Training**: The `RandomForestClassifier` is trained on the `urls.csv` dataset using these extracted features.
3.  **Prediction**: The trained model (`url_detector_model.pkl`) is loaded by the Streamlit application. When a user submits a URL, the app extracts its features and uses the model to predict its classification.

## 🛠️ Technologies Used

  * **Python**
  * **Scikit-learn**: For the machine learning model (`RandomForestClassifier`).
  * **Pandas**: For data manipulation and reading the CSV file.
  * **Streamlit**: For creating the interactive web application.
  * **Joblib**: For saving and loading the trained model.

## 📁 File Descriptions

  * **`app.py`**: Contains the code for the Streamlit web application that serves the model.
  * **`train_url_detector.py`**: The script used to extract features, train the `RandomForestClassifier`, and save the model to a `.pkl` file.
  * **`urls.csv`**: The dataset containing labeled URLs used for training and testing the model.
  * **`url_detector_model.pkl`**: The pre-trained, serialized machine learning model file.

## 📦 Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    Create a `requirements.txt` file with the following content:

    ```
    pandas
    scikit-learn
    streamlit
    ```

    Then, run the command:

    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Usage

### Running the Web Application

1.  Ensure you have the trained model file `url_detector_model.pkl` in the root directory.
2.  Launch the Streamlit application by running the following command in your terminal:
    ```bash
    streamlit run app.py
    ```
3.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

### Re-training the Model

If you modify the dataset or want to retrain the model, you can run the training script:

```bash
python train_url_detector.py
```

This will train a new model on the `urls.csv` data and overwrite the existing `url_detector_model.pkl` file.

## 📊 Model Performance

The model's performance is evaluated using accuracy and a detailed classification report. After running `train_url_detector.py`, the console will display the results. You can paste your results here.

*(Example Placeholder)*

```
Accuracy: 0.96

              precision    recall  f1-score   support

   malicious       0.98      0.97      0.97       750
    phishing       0.95      0.96      0.95       980
        safe       0.97      0.96      0.96      1250

    accuracy                           0.96      2980
   macro avg       0.96      0.96      0.96      2980
weighted avg       0.96      0.96      0.96      2980
```

-----

## 👨‍💻 Developed By

Muhammad Jarrar Hassan
