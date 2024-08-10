from flask import Flask, request, render_template, send_file, redirect, url_for
import joblib
import numpy as np
import xgboost as xgb
import pandas as pd
from datetime import datetime
import boto3
from botocore.exceptions import NoCredentialsError
import os
import logging
from sklearn.metrics import confusion_matrix
import psycopg2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Database configuration
DATABASE_URL = os.getenv('CUSTOM_DATABASE_URL')

def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

# Load the saved model and selected features from local storage
model_path = 'xgboost_final_modelMTMR.json'
features_path = 'final_featuresMTMR.pkl'

# Load the model using XGBoost's load_model method
model = xgb.XGBClassifier()
model.load_model(model_path)

final_features = joblib.load(features_path)

# Route to render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Extract form data
        form_data = request.form.to_dict()
        logger.info(f"Received form data: {form_data}")

        mrn = form_data['mrn']
        date = form_data['date']

        gender = form_data['gender'].upper()
        gender = 0 if gender == 'F' else 1

        age = float(form_data['age'])
        log_age = np.log(age + 1)

        hypertension = form_data['hypertension'].lower()
        hypertension = 1 if hypertension == 'yes' else 0

        a1c = float(form_data['a1c'])
        log_a1c = np.log(a1c + 1)

        duration = float(form_data['duration'])
        log_duration = np.log(duration + 1)

        age_diagnosis = age - duration

        therapy = form_data['therapy'].lower()
        therapy_combined = 1 if therapy == 'combined' else 0
        therapy_tablets = 1 if therapy == 'tablets' else 0
        therapy_insulin = 1 if therapy == 'insulin' else 0

        cardiac = form_data['cardiac'].lower()
        cardiac = 1 if cardiac == 'yes' else 0

        neuropathy = form_data['neuropathy'].lower()
        neuropathy = 1 if neuropathy == 'yes' else 0

        renal = form_data['renal'].lower()
        renal = 1 if renal == 'yes' else 0

        smoking = form_data['smoking'].lower()
        smoking_current = 1 if smoking == 'current' else 0
        smoking_never = 1 if smoking == 'never' else 0
        smoking_past = 1 if smoking == 'past' else 0

        log_age_hypertension = log_age * hypertension
        log_age_a1c = log_age * log_a1c
        log_duration_a1c = log_duration * log_a1c
        cardiac_neuropathy = cardiac * neuropathy

        features = np.array([gender, log_age, age_diagnosis, hypertension, log_a1c, log_duration,
                             cardiac, neuropathy, renal, therapy_combined, therapy_tablets, therapy_insulin,
                             smoking_current, smoking_never, smoking_past,
                             log_age_hypertension, log_age_a1c, log_duration_a1c, cardiac_neuropathy]).reshape(1, -1)

        selected_features_values = features[:, :len(final_features)]  # Ensure the feature length matches the model

        prediction = model.predict(selected_features_values)
        prediction_proba = float(model.predict_proba(selected_features_values)[:, 1][0])

        prediction_label = 'MTMR' if prediction[0] == 1 else 'Non-MTMR'

        # Save to database
        cursor.execute('''
            INSERT INTO patient (mrn, date, gender, age, age_diagnosis, hypertension, a1c, duration, 
            therapy_combined, therapy_tablets, therapy_insulin, cardiac, neuropathy, renal, 
            smoking_current, smoking_never, smoking_past, prediction, probability, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        ''', (mrn, date, gender, age, age_diagnosis, hypertension, a1c, duration, therapy_combined, 
              therapy_tablets, therapy_insulin, cardiac, neuropathy, renal, smoking_current, smoking_never, 
              smoking_past, prediction_label, prediction_proba, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        new_id = cursor.fetchone()[0]
        conn.commit()

        return render_template('result.html', mrn=mrn, date=date, prediction=prediction_label, 
                               probability=prediction_proba, prediction_id=new_id)
    except Exception as e:
        conn.rollback()
        logger.error(f"Error during prediction: {e}")
        return "An error occurred during prediction. Please check the server logs for more details."
    finally:
        cursor.close()
        conn.close()

# Route to update ground truth
@app.route('/update_ground_truth', methods=['POST'])
def update_ground_truth():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        mrn = request.form['mrn']
        ground_truth = request.form['ground_truth']
        prediction_id = request.form['prediction_id']

        cursor.execute('SELECT * FROM patient WHERE id = %s;', (prediction_id,))
        patient = cursor.fetchone()

        if patient:
            cursor.execute('UPDATE patient SET ground_truth = %s WHERE id = %s;', (ground_truth, prediction_id))
            conn.commit()
            return redirect(f'/prediction_result/{prediction_id}')
        else:
            logger.error(f"Prediction not found for id: {prediction_id}")
            return "Prediction not found"
    except Exception as e:
        conn.rollback()
        logger.error(f"Error updating ground truth: {e}")
        return "An error occurred while updating ground truth. Please check the server logs for more details."
    finally:
        cursor.close()
        conn.close()

# Route to view prediction result
@app.route('/prediction_result/<int:prediction_id>')
def prediction_result(prediction_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM patient WHERE id = %s;', (prediction_id,))
        patient = cursor.fetchone()

        if patient:
            return render_template('result.html', mrn=patient[1], date=patient[2], 
                                   prediction=patient[16], probability=patient[17], prediction_id=prediction_id)
        else:
            return "Prediction not found"
    except Exception as e:
        logger.error(f"Error fetching prediction result: {e}")
        return "An error occurred. Please check the server logs for more details."
    finally:
        cursor.close()
        conn.close()

# Route to view all patients
@app.route('/patients')
def view_patients():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM patient;')
        patients = cursor.fetchall()

        # Calculate metrics
        cursor.execute('SELECT * FROM patient WHERE ground_truth IS NOT NULL;')
        results = cursor.fetchall()
        if results:
            y_true = [1 if patient[18] == 'MTMR' else 0 for patient in results]
            y_pred = [1 if patient[16] == 'MTMR' else 0 for patient in results]

            tn, fp, fn, tp = 0, 0, 0, 0  # Initialize default values
            if len(set(y_true)) > 1 and len(set(y_pred)) > 1:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            elif len(set(y_true)) == 1 and len(set(y_pred)) == 1:
                if y_true[0] == 1:
                    tp = y_true.count(1)
                else:
                    tn = y_true.count(0)

            sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
        else:
            sensitivity = specificity = accuracy = None

        return render_template('patients.html', patients=patients, sensitivity=sensitivity, 
                               specificity=specificity, accuracy=accuracy)
    except Exception as e:
        logger.error(f"Error viewing patients: {e}")
        return "An error occurred. Please check the server logs for more details."
    finally:
        cursor.close()
        conn.close()

# Route to delete a patient
@app.route('/delete_patient/<int:patient_id>')
def delete_patient(patient_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('DELETE FROM patient WHERE id = %s;', (patient_id,))
        conn.commit()
        return redirect(url_for('view_patients'))
    except Exception as e:
        conn.rollback()
        logger.error(f"Error deleting patient: {e}")
        return "An error occurred while deleting the patient. Please check the server logs for more details."
    finally:
        cursor.close()
        conn.close()

# Route to download data as CSV
@app.route('/download')
def download_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM patient;')
        patients = cursor.fetchall()

        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(patients, columns=columns)
        df.to_csv('patients_data.csv', index=False)
        return send_file('patients_data.csv', as_attachment=True)
    except Exception as e:
        logger.error(f"Error during CSV download: {e}")
        return "An error occurred during CSV download. Please check the server logs for more details."
    finally:
        cursor.close()
        conn.close()

# Route to backup the database to Amazon S3
def backup_to_s3():
    s3 = boto3.client('s3', aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'), 
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))

    try:
        s3.upload_file('patients.db', 'your-bucket-name', 'patients_backup.db')
        logger.info("Backup successful")
    except FileNotFoundError:
        logger.error("The file was not found")
    except NoCredentialsError:
        logger.error("Credentials not available")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == '__main__':
    # Initialize the database (ensure tables are created if they don't exist)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patient (
            id SERIAL PRIMARY KEY,
            mrn VARCHAR(64) UNIQUE NOT NULL,
            date VARCHAR(64),
            gender INTEGER,
            age FLOAT,
            age_diagnosis FLOAT,
            hypertension INTEGER,
            a1c FLOAT,
            duration FLOAT,
            therapy_combined INTEGER,
            therapy_tablets INTEGER,
            therapy_insulin INTEGER,
            cardiac INTEGER,
            neuropathy INTEGER,
            renal INTEGER,
            smoking_current INTEGER,
            smoking_never INTEGER,
            smoking_past INTEGER,
            prediction VARCHAR(64),
            probability FLOAT,
            ground_truth VARCHAR(64),
            timestamp VARCHAR(64)
        );
    ''')
    conn.commit()
    cursor.close()
    conn.close()
    
    app.run(debug=True)