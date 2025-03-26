from flask import Flask, render_template, request
from dotenv import load_dotenv

import numpy as np
import pickle
import boto3
import os

app = Flask(__name__)


from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read AWS credentials from environment
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")  # Optional
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_KEY = os.getenv("S3_KEY")
LOCAL_MODEL_PATH = "logistic_regrssion_model.pkl"

def download_model_from_s3():
    """Download model from S3 if not available locally."""
    if not os.path.exists(LOCAL_MODEL_PATH):
        s3 = boto3.client('s3',
                          aws_access_key_id=AWS_ACCESS_KEY_ID,
                          aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                          aws_session_token=AWS_SESSION_TOKEN,
                          region_name=AWS_REGION)
        s3.download_file(S3_BUCKET, S3_KEY, LOCAL_MODEL_PATH)
        print("Model downloaded from S3!")

# Download model before starting the app
download_model_from_s3()

# Load the trained model
with open("logistic_regrssion_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)  # Assuming it's a single model
    
test_input = np.array([[12282, 2, 1000, 1, 11.14, 0.08, 1]])  # Sample row
print("Prediction:", model.predict(test_input))
    
# Debugging: Print the type of the loaded model
print("Loaded model type:", type(model))

# @app.route("/", methods=["GET", "POST"])
# def home():
#     if request.method == "POST":
#         try:
#             print("Received Form Data:", request.form)  # Debugging
            
#             person_income = float(request.form["person_income"])
#             home_ownership = int(request.form["home_ownership"])
#             loan_amnt = float(request.form["loan_amnt"])
#             loan_intent = int(request.form["loan_intent"])
#             loan_int_rate = float(request.form["loan_int_rate"])
#             loan_percent_income = float(request.form["loan_percent_income"])
#             previous_loan_defaults_on_file = int(request.form["previous_loan_defaults_on_file"])

#             input_data = np.array([[person_income, home_ownership, loan_amnt, 
#                                     loan_intent, loan_int_rate, loan_percent_income, 
#                                     previous_loan_defaults_on_file]])

#             prediction = model.predict(input_data)
#             prediction_text = "Approved" if prediction[0] == 1 else "Rejected"

#         except Exception as e:
#             prediction_text = f"Error: {str(e)}"

#         return render_template("index.html", prediction=prediction_text)

#     return render_template("index.html")

# @app.route("/", methods=["GET", "POST"])
# def home():
#     if request.method == "POST":
#         try:
#             person_income = float(request.form["person_income"])
#             home_ownership = int(request.form["home_ownership"])
#             loan_amnt = float(request.form["loan_amnt"])
#             loan_intent = int(request.form["loan_intent"])
#             loan_int_rate = float(request.form["loan_int_rate"])
#             loan_percent_income = float(request.form["loan_percent_income"])
#             previous_loan_defaults_on_file = int(request.form["previous_loan_defaults_on_file"])

#             input_data = np.array([[person_income, home_ownership, loan_amnt, 
#                                     loan_intent, loan_int_rate, loan_percent_income, 
#                                     previous_loan_defaults_on_file]])

#             # Get the probability of each class (Rejected, Approved)
#             probas = model.predict_proba(input_data)

#             # Debugging: Print the probabilities
#             print("Prediction probabilities:", probas)

#             # Adjust the threshold for classifying as "Approved"
#             threshold = 0.2  # Lower threshold to be more lenient for "Approved"
#             prediction = (probas[:, 1] > threshold).astype(int)

#             # Determine the prediction
#             prediction_text = "Approved" if prediction[0] == 1 else "Rejected"

#         except Exception as e:
#             prediction_text = f"Error: {str(e)}"

#         return render_template("index.html", prediction=prediction_text)

#     return render_template("index.html")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            person_income = float(request.form["person_income"])
            home_ownership = int(request.form["home_ownership"])
            loan_amnt = float(request.form["loan_amnt"])
            loan_intent = int(request.form["loan_intent"])
            loan_int_rate = float(request.form["loan_int_rate"])
            loan_percent_income = float(request.form["loan_percent_income"])
            previous_loan_defaults_on_file = int(request.form["previous_loan_defaults_on_file"])

            input_data = np.array([[person_income, home_ownership, loan_amnt, 
                                    loan_intent, loan_int_rate, loan_percent_income, 
                                    previous_loan_defaults_on_file]])

            # Get the probability of each class (Rejected, Approved)
            probas = model.predict_proba(input_data)

            # Debugging: Print the probabilities
            print("Prediction probabilities:", probas)

            # Adjust the threshold for classifying as "Approved"
            threshold = 0.1  # Lower threshold to be more lenient for "Approved"
            prediction = (probas[:, 1] > threshold).astype(int)

            # Determine the prediction
            prediction_text = "Approved" if prediction[0] == 1 else "Rejected"

        except Exception as e:
            prediction_text = f"Error: {str(e)}"

        return render_template("index.html", prediction=prediction_text)

    # Test the model manually with suggested input data for debugging
    test_input = np.array([[70000, 1, 25000, 0, 5.0, 0.2, 0]])  # Using the input data you provided
    probas = model.predict_proba(test_input)
    print("Manual Test Prediction Probabilities:", probas)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)