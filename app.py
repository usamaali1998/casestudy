from flask import Flask, request, render_template, send_file
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import traceback
import os

matplotlib.use('Agg')  # Use non-GUI backend for Matplotlib

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
    try:
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        # Load model
        model_name = request.form.get('model')
        model = models.get(model_name)
        if not model:
            return "Invalid model selected"

        # Read the file
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith('.xls') or file.filename.endswith('.xlsx'):
            data = pd.read_excel(file, engine='openpyxl')
        else:
            return "Unsupported file format. Please upload a CSV or Excel file."

        # Preprocess and predict
        data = preprocess_data(data)
        predictions = make_predictions(data, model)

        # Save predictions to a file
        predictions_file_path = os.path.join('static', 'predictions.xlsx')
        predictions.to_excel(predictions_file_path, index=False)

        # Generate graphs for all parameters
        if not os.path.exists('static'):
            os.makedirs('static')

        graph_paths = []
        for column in columns:
            graph_path = f'static/{column}_vs_predictions.png'
            plt.figure(figsize=(8, 6))
            plt.scatter(data[column], predictions['Predictions'], alpha=0.6)
            plt.xlabel(column)
            plt.ylabel('Predictions')
            plt.title(f'Predictions vs. {column}')
            plt.savefig(graph_path)
            plt.close()
            graph_paths.append(graph_path)

        return render_template('result.html', graph_paths=graph_paths, predictions_download="predictions.xlsx")
    except Exception as e:
        print(traceback.format_exc())
        return str(e)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_name = request.form.get('model')
        model = models.get(model_name)
        record = {col: float(request.form[col]) for col in columns}
        single_record = pd.DataFrame([record])
        prediction = model.predict(single_record)[0]

        output = BytesIO()
        single_record['Prediction'] = [prediction]
        single_record.to_excel(output, index=False)
        output.seek(0)
        return send_file(output, download_name="manual_prediction.xlsx", as_attachment=True)
    except Exception as e:
        print(traceback.format_exc())
        return str(e)

def preprocess_data(data):
    data = data.copy()  # Create a deep copy to avoid SettingWithCopyWarning
    
    # Replace inconsistent values in 'Item_Fat_Content'
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({
        'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular', 'regular': 'Regular'
    })

    # Calculate 'O_Years' from 'Outlet_Establishment_Year'
    data.loc[:, 'O_Years'] = 2024 - data['Outlet_Establishment_Year']
    
    # Handle missing values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')  # Use mean to replace NaN values
    numeric_columns = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'O_Years']
    data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
    
    # Encode categorical columns
    le = LabelEncoder()
    data.loc[:, 'I_Fate_Content'] = le.fit_transform(data['Item_Fat_Content'])
    data.loc[:, 'I_Type'] = le.fit_transform(data['Item_Type'])
    data.loc[:, 'O_Size'] = le.fit_transform(data['Outlet_Size'])
    data.loc[:, 'O_Location_Type'] = le.fit_transform(data['Outlet_Location_Type'])
    data.loc[:, 'O_Type'] = le.fit_transform(data['Outlet_Type'])

    # Select only the columns used for prediction
    return data[columns]

def make_predictions(data, model):
    predictions = model.predict(data)
    data['Predictions'] = predictions
    return data

if __name__ == '__main__':
    app.run(debug=True)
