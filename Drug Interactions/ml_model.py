from flask import jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
# For encoding data in numbers instead of string
data = pd.read_csv('Drug_interaction.csv')


label_encoders = {}
for column in ['Drug1', 'Drug2', 'Interaction_Type', 'Severity', 'Mechanism', 'Clinical_Relevance']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split the data into features and targets
X = data[['Drug1', 'Drug2']]
y_severity = data['Severity']
y_mechanism = data['Mechanism']
y_clinical_relevance = data['Clinical_Relevance']

# Split the data into training and testing sets
X_train, X_test, y_severity_train, y_severity_test = train_test_split(X, y_severity, test_size=0.2, random_state=42)
X_train, X_test, y_mechanism_train, y_mechanism_test = train_test_split(X, y_mechanism, test_size=0.2, random_state=42)
X_train, X_test, y_clinical_relevance_train, y_clinical_relevance_test = train_test_split(X, y_clinical_relevance, test_size=0.2, random_state=42)

# Train the model for severity prediction
severity_model = RandomForestClassifier(random_state=42)
severity_model.fit(X_train, y_severity_train)

# Train the model for mechanism prediction
mechanism_model = RandomForestClassifier(random_state=42)
mechanism_model.fit(X_train, y_mechanism_train)

# Train the model for clinical relevance prediction
clinical_relevance_model = RandomForestClassifier(random_state=42)
clinical_relevance_model.fit(X_train, y_clinical_relevance_train)

# Predict and evaluate the models
y_severity_pred = severity_model.predict(X_test)
y_mechanism_pred = mechanism_model.predict(X_test)
y_clinical_relevance_pred = clinical_relevance_model.predict(X_test)

print("Severity Model Accuracy:", accuracy_score(y_severity_test, y_severity_pred))
print("Mechanism Model Accuracy:", accuracy_score(y_mechanism_test, y_mechanism_pred))
print("Clinical Relevance Model Accuracy:", accuracy_score(y_clinical_relevance_test, y_clinical_relevance_pred))

joblib.dump(severity_model, './pickleFiles/severity_model.pkl')
joblib.dump(mechanism_model, './pickleFiles/mechanism_model.pkl')
joblib.dump(clinical_relevance_model, './pickleFiles/clinical_relevance_model.pkl')

for column, le in label_encoders.items():
    joblib.dump(le, f'./pickleFiles/label_encoder_{column}.pkl')


def identify(drug1, drug2):
    try:
        input_data = [[drug1,drug2]]
        severity = severity_model.predict(input_data)
        mechanism = mechanism_model.predict(input_data)
        clinical_relevance = clinical_relevance_model.predict(input_data)
        
        print(severity)
        print(mechanism)
        print(clinical_relevance)
        response = {
        'severity': severity,
        'mechanism': mechanism,
        'clinical_relevance': clinical_relevance
        }
    
        return jsonify(response)
    except Exception as e:
        print(e)
        return "error"

