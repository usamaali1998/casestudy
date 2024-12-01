from flask import Flask, request, render_template, send_file
import pandas as pd
import joblib
from io import BytesIO
import traceback
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Load the trained models
models = {
    'linear': joblib.load('linear_model.pkl'),
    'random_forest': joblib.load('random_forest_model.pkl'),
    'gradient_boosting': joblib.load('gradient_boosting_model.pkl')
}

# Define the columns used for prediction
columns = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'O_Years', 'I_Fate_Content', 'I_Type', 'O_Size', 'O_Location_Type', 'O_Type']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        try:
            model_name = request.form.get('model')
            model = models.get(model_name)
            if not model:
                return "Invalid model selected"

            filename = file.filename
            if filename.endswith('.csv'):
                data = pd.read_csv(file)
            elif filename.endswith('.xls') or filename.endswith('.xlsx'):
                data = pd.read_excel(file, engine='openpyxl')
            else:
                return "Unsupported file format. Please upload a CSV or Excel file."

            # Preprocess the data
            data = preprocess_data(data)
            predictions = make_predictions(data, model)
            output = BytesIO()
            predictions.to_excel(output, index=False)
            output.seek(0)
            return send_file(output, download_name="predictions.xlsx", as_attachment=True)
        except Exception as e:
            print(traceback.format_exc())
            return str(e)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_name = request.form.get('model')
        model = models.get(model_name)
        if not model:
            return "Invalid model selected"
        
        record = request.form.to_dict()
        record = {col: float(record[col]) for col in columns}
        single_record = pd.DataFrame([record], columns=columns)
        prediction = model.predict(single_record)[0]
        return f"Predicted value: {prediction}"
    except Exception as e:
        print(traceback.format_exc())
        return str(e)

def preprocess_data(data):
    # Handle missing values
    data = data.dropna()

    # Replace inconsistent values in 'Item_Fat_Content'
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({
        'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat', 'regular': 'Regular'
    })

    # Calculate 'O_Years' from 'Outlet_Establishment_Year'
    data['O_Years'] = 2024 - data['Outlet_Establishment_Year']

    # Encode categorical columns
    le = LabelEncoder()
    data['I_Fate_Content'] = le.fit_transform(data['Item_Fat_Content'])
    data['I_Type'] = le.fit_transform(data['Item_Type'])
    data['O_Size'] = le.fit_transform(data['Outlet_Size'])
    data['O_Location_Type'] = le.fit_transform(data['Outlet_Location_Type'])
    data['O_Type'] = le.fit_transform(data['Outlet_Type'])

    # Select only the necessary columns
    data = data[columns]

    return data

def make_predictions(data, model):
    predictions = model.predict(data)
    data['Predictions'] = predictions
    return data

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
