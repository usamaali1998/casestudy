from flask import Flask, request, render_template, send_file
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import joblib
from io import BytesIO
import traceback
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

def generate_plot():
    models = ["Linear Regression", "Random Forest", "Gradient Boosting"]

    # Metrics
    rmse_train = [1100.55, 433.10, 1037.51]
    mae_train = [814.06, 317.45, 767.64]
    r2_train = [0.4648, 0.9171, 0.5243]

    rmse_test = [1041.43, 1102.07, 1055.36]
    mae_test = [765.03, 812.00, 776.09]
    r2_test = [0.4981, 0.4380, 0.4846]

    x = np.arange(len(models))
    width = 0.3

    # Store images in a dictionary
    images = {}

    # Plot RMSE Comparison
    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, rmse_train, width, label="Train RMSE")
    plt.bar(x + width/2, rmse_test, width, label="Test RMSE")
    plt.xlabel("Models")
    plt.ylabel("RMSE")
    plt.title("Root Mean Squared Error Comparison")
    plt.xticks(x, models)
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    images['rmse'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # Plot MAE Comparison
    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, mae_train, width, label="Train MAE")
    plt.bar(x + width/2, mae_test, width, label="Test MAE")
    plt.xlabel("Models")
    plt.ylabel("MAE")
    plt.title("Mean Absolute Error Comparison")
    plt.xticks(x, models)
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    images['mae'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # Plot R² Score Comparison
    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, r2_train, width, label="Train R²")
    plt.bar(x + width/2, r2_test, width, label="Test R²")
    plt.xlabel("Models")
    plt.ylabel("R² Score")
    plt.title("R-Squared Score Comparison")
    plt.xticks(x, models)
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    images['r2'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return images

# Load the trained models
models = {
    'linear': joblib.load('linear_model.pkl'),
    'random_forest': joblib.load('random_forest_model.pkl'),
    'gradient_boosting': joblib.load('gradient_boosting_model.pkl')
}

# Define the columns used for prediction (model-trained names)
columns = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'O_Years', 'I_Fate_Content', 'I_Type', 'O_Size', 'O_Location_Type', 'O_Type']

# Load the dataset
cleaned_data = pd.read_csv('Unique_Mapped_Product_Names.csv')

# Define the mapping of Item_Type to Physical Product Names
item_name_mapping = {
    "Baking Goods": "All-Purpose Flour, Baking Powder, Cocoa Powder",
    "Breads": "Whole Wheat Bread, Multigrain Bread, Baguette",
    "Breakfast": "Oats, Cornflakes, Granola",
    "Canned": "Canned Beans, Canned Corn, Canned Tomatoes",
    "Dairy": "Milk, Butter, Cheese",
    "Frozen Foods": "Frozen Pizza, Frozen Vegetables, Ice Cream",
    "Fruits and Vegetables": "Apples, Carrots, Spinach",
    "Hard Drinks": "Whiskey, Vodka, Rum",
    "Health and Hygiene": "Toothpaste, Shampoo, Soap",
    "Household": "Detergent, Dish Soap, Paper Towels",
    "Meat": "Chicken Breast, Beef Steak, Lamb Chops",
    "Others": "Miscellaneous Grocery Items",
    "Seafood": "Salmon, Shrimp, Tuna",
    "Snack Foods": "Potato Chips, Popcorn, Chocolate Bars",
    "Soft Drinks": "Cola, Lemonade, Orange Juice",
    "Starchy Foods": "Rice, Pasta, Potatoes"
}

# Map Physical_Product_Name from Item_Type and extract only a single product
cleaned_data["Physical_Product_Name"] = cleaned_data["Item_Type"].map(item_name_mapping)
cleaned_data["Physical_Product_Name"] = cleaned_data["Physical_Product_Name"].apply(lambda x: x.split(", ")[0] if isinstance(x, str) else x)

# Convert dataframe to a list of tuples containing (Item_Identifier, Physical_Product_Name)
item_identifiers = cleaned_data[["Item_Identifier", "Physical_Product_Name","Item_Type"]].dropna().drop_duplicates().values.tolist()




@app.route('/')
def home():
     images = generate_plot()
     return render_template('index.html', item_identifiers=item_identifiers ,images=images )

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
        # Map user-friendly names to model-compatible names
        record = {
            'Item_Weight': float(record['Item_Weight']),
            'Item_Visibility': float(record['Item_Visibility']),
            'Item_MRP': float(record['Item_MRP']),
            'O_Years': float(record['O_Years']),
            'I_Fate_Content': float(record['Item_Fat_Content']),  # Mapping
            'I_Type': float(record['Item_Type']),  # Mapping
            'O_Size': float(record['Outlet_Size']),  # Mapping
            'O_Location_Type': float(record['Outlet_Location_Type']),  # Mapping
            'O_Type': float(record['Outlet_Type']),  # Mapping
        }
        single_record = pd.DataFrame([record], columns=columns)
        prediction = model.predict(single_record)[0]
        return f"The Predicted Sales value is : {prediction}"
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
