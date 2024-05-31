from flask import Flask, request, jsonify,request, redirect, url_for
import joblib
from ml_model import identify
from flask import render_template
import pandas as pd
import requests
import json
import http.client
conn = http.client.HTTPSConnection("weatherapi-com.p.rapidapi.com")
app = Flask(__name__)

# Load models and encoders
severity_model = joblib.load('./pickleFiles/severity_model.pkl')
mechanism_model = joblib.load('./pickleFiles/mechanism_model.pkl')
clinical_relevance_model = joblib.load('./pickleFiles/clinical_relevance_model.pkl')

label_encoders = {}
for column in ['Drug1', 'Drug2', 'Interaction_Type', 'Severity', 'Mechanism', 'Clinical_Relevance']:
    label_encoders[column] = joblib.load(f'./pickleFiles/label_encoder_{column}.pkl')

def encode_drug_names(drug1, drug2):
    try:
        drug1_encoded = label_encoders['Drug1'].transform([drug1])[0]
        drug2_encoded = label_encoders['Drug2'].transform([drug2])[0]
        return drug1_encoded, drug2_encoded
    except Exception as e:
        print(e)
        return None, None



@app.route("/")
def hello_world():
    return render_template('home.html')
    # return "<p>Hello, World!</p>"
@app.route('/identify', methods=['POST'])
def get_recommendation():
    # data = request.json
    print("request object is:")
    print(request.form['drug1'])
    print(request.form['drug2'])
    drug1 = request.form['drug1']
    drug2 = request.form['drug2']
    if not drug1 or not drug2:
        return jsonify({'error': 'Missing drug names'}), 400
    
    # Encode the drug names
    drug1_encoded, drug2_encoded = encode_drug_names(drug1, drug2)
    
    if drug1_encoded is None or drug2_encoded is None:
        return jsonify({'error': 'Invalid drug names'}), 400
    # Get recommendation from your recommendation function
    try:    
        
    
        input_data = pd.DataFrame([[drug1_encoded, drug2_encoded]], columns=['Drug1', 'Drug2'])
    
    # Predict severity, mechanism, and clinical relevance
        severity_pred = severity_model.predict(input_data)[0]
        mechanism_pred = mechanism_model.predict(input_data)[0]
        clinical_relevance_pred = clinical_relevance_model.predict(input_data)[0]
    
    # Decode the predictions
        severity = label_encoders['Severity'].inverse_transform([severity_pred])[0]
        mechanism = label_encoders['Mechanism'].inverse_transform([mechanism_pred])[0]
        clinical_relevance = label_encoders['Clinical_Relevance'].inverse_transform([clinical_relevance_pred])[0]
    
        response = {
            'severity': severity,
            'mechanism': mechanism,
            'clinical_relevance': clinical_relevance
        }
        print("response is")
        print(response)
        return jsonify(response)
        # return render_template('home.html', drug1 = drug1, drug2 = drug2)
        # return jsonify({'crop': recommended_crop})
    except Exception as e:
        print(e)


    

@app.route("/login")
def login():
    return render_template('login.html')


if __name__ == '__main__':
    app.run(debug=True)
