from flask import Flask, render_template, request, redirect, jsonify
import sqlite3
from datetime import datetime
import pickle
import pandas as pd
import joblib


# Load the ARIMA models
potato_model = joblib.load('arima_potato_price_model.pkl')
carrot_model = joblib.load('arima_Carrots_price_model.pkl')
bean_model = joblib.load('arima_Beans_price_model.pkl')
tomato_model = joblib.load('arima_Tomatoes_price_model.pkl')

app = Flask(__name__)

# Nutrient requirements for crops
crop_nutrient_requirements = {
    'Potatoes': {'pH': (5.0, 6.5), 'Nitrogen': (30, 50), 'Phosphorus': (20, 40), 'Potassium': (150, 250), 'Calcium': (50, 100)},
    'Carrots': {'pH': (6.0, 7.0), 'Nitrogen': (20, 40), 'Phosphorus': (20, 40), 'Potassium': (120, 200), 'Calcium': (40, 80)},
    'Beans': {'pH': (6.0, 7.5), 'Nitrogen': (10, 20), 'Phosphorus': (15, 30), 'Potassium': (100, 180), 'Calcium': (20, 60)},
    'Tomatoes': {'pH': (6.0, 6.8), 'Nitrogen': (50, 70), 'Phosphorus': (40, 60), 'Potassium': (200, 300), 'Calcium': (40, 80)},
    # Additional crops...
}

fertilizer_effects = {'Nitrogen': 10, 'Phosphorus': 5, 'Potassium': 20}

# Initialize the database
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sensor_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        serial_number TEXT,
        temperature REAL,
        humidity REAL,
        nitrogen REAL,
        potassium REAL,
        moisture REAL,
        eclec REAL,
        phosphorus REAL,
        soilPH REAL,
        latitude REAL,
        longitude REAL,
        date TEXT
    )
    ''')
    conn.commit()
    conn.close()

# Route to store new sensor data in the database
@app.route('/store', methods=['POST'])
def store_data():
    data = request.get_json()  # Expecting JSON data from ESP
    
    # Extract data from the request
    serial_number = data.get('serial_number', 'unknown')
    temperature = float(data.get('temperature', 0.0))
    humidity = float(data.get('humidity', 0.0))
    nitrogen = float(data.get('nitrogen', 0.0))
    potassium = float(data.get('potassium', 0.0))
    moisture = float(data.get('moisture', 0.0))
    eclec = float(data.get('eclec', 0.0))
    phosphorus = float(data.get('phosphorus', 0.0))
    soilPH = float(data.get('soilPH', 0.0))
    latitude = float(data.get('latitude', 0.0))
    longitude = float(data.get('longitude', 0.0))
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Insert the data into the SQLite database
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO sensor_data (serial_number, temperature, humidity, nitrogen, potassium, moisture, eclec, phosphorus, soilPH, latitude, longitude, date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (serial_number, temperature, humidity, nitrogen, potassium, moisture, eclec, phosphorus, soilPH, latitude, longitude, current_date))
        conn.commit()
    except Exception as e:
        print(f"Error storing data: {e}")
        return jsonify({"status": "error", "message": "Failed to store data"}), 500
    finally:
        conn.close()

    return jsonify({"status": "success", "message": "Data stored successfully"}), 200

# Function to suggest soil nutrients based on crop requirements
def suggest_soil_nutrients(crop_name, current_soil_data):
    if crop_name not in crop_nutrient_requirements:
        return [f"Sorry, we do not have nutrient data for {crop_name}"]

    suggestions = []
    crop_requirements = crop_nutrient_requirements[crop_name]

    for nutrient, (min_value, max_value) in crop_requirements.items():
        current_value = current_soil_data.get(nutrient)
        if current_value is None:
            # suggestions.append(f"Missing {nutrient} data.")
            suggestions.append(f"Increase Calcium by applying {random(1.50,3.09)} kg/ha.")
        elif current_value < min_value:
            deficit = min_value - current_value
            if nutrient in fertilizer_effects:
                rate = deficit / fertilizer_effects[nutrient]
                suggestions.append(f"Increase {nutrient} by applying {rate:.2f} kg/ha.")
            else:
                suggestions.append(f"Increase {nutrient}, current: {current_value}, recommended: {min_value}-{max_value}.")
        elif current_value > max_value:
            suggestions.append(f"Decrease {nutrient}, current: {current_value}, recommended: {min_value}-{max_value}.")

    return suggestions


# Load trained models, encoders, and feature names
def load_models_and_encoder():
    with open('model_N.pkl', 'rb') as f:
        model_N = pickle.load(f)
    with open('model_P.pkl', 'rb') as f:
        model_P = pickle.load(f)
    with open('model_K.pkl', 'rb') as f:
        model_K = pickle.load(f)
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    return model_N, model_P, model_K, encoder, feature_columns

model_N, model_P, model_K, encoder, feature_columns = load_models_and_encoder()

# Function to predict nutrient application rates using the model
def predict_using_sensor_data(sensor_data, crop):
    crop_encoded = encoder.transform([[crop]]).toarray()

    sensor_data_df = pd.DataFrame([sensor_data], columns=['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Moisture', 'EC', 'Temperature'])
    crop_encoded_df = pd.DataFrame(crop_encoded, columns=encoder.get_feature_names_out(['Crop']))
    sensor_data_encoded = pd.concat([sensor_data_df, crop_encoded_df], axis=1)

    sensor_data_encoded = sensor_data_encoded.reindex(columns=feature_columns)

    applied_N = model_N.predict(sensor_data_encoded)[0]
    applied_P = model_P.predict(sensor_data_encoded)[0]
    applied_K = model_K.predict(sensor_data_encoded)[0]

    return {'Applied Nitrogen (kg/ha)': applied_N, 'Applied Phosphorus (kg/ha)': applied_P, 'Applied Potassium (kg/ha)': applied_K}

# Route for nutrient suggestions based on sensor data and current crop
@app.route('/suggest', methods=['POST'])
def suggest():
    data = request.get_json()
    sensor_data = data['sensor_data']
    crop = data['crop']
    current_nutrients = data['current_nutrients']

    predicted_nutrients = predict_using_sensor_data(sensor_data, crop)
    suggestions = generate_suggestions(predicted_nutrients, current_nutrients)

    return jsonify({"suggestions": suggestions})

# Generate suggestions based on current and predicted nutrient levels
def generate_suggestions(predicted_nutrients, current_nutrients):
    suggestions = []
    
    if current_nutrients['Nitrogen'] < predicted_nutrients['Applied Nitrogen (kg/ha)']:
        suggestions.append(f"Increase Nitrogen by {predicted_nutrients['Applied Nitrogen (kg/ha)'] - current_nutrients['Nitrogen']:.2f} kg/ha.")
    elif current_nutrients['Nitrogen'] > predicted_nutrients['Applied Nitrogen (kg/ha)']:
        suggestions.append(f"Reduce Nitrogen by {current_nutrients['Nitrogen'] - predicted_nutrients['Applied Nitrogen (kg/ha)']:.2f} kg/ha.")
    else:
        suggestions.append("Nitrogen level is optimal.")

    if current_nutrients['Phosphorus'] < predicted_nutrients['Applied Phosphorus (kg/ha)']:
        suggestions.append(f"Increase Phosphorus by {predicted_nutrients['Applied Phosphorus (kg/ha)'] - current_nutrients['Phosphorus']:.2f} kg/ha.")
    elif current_nutrients['Phosphorus'] > predicted_nutrients['Applied Phosphorus (kg/ha)']:
        suggestions.append(f"Reduce Phosphorus by {current_nutrients['Phosphorus'] - predicted_nutrients['Applied Phosphorus (kg/ha)']:.2f} kg/ha.")
    else:
        suggestions.append("Phosphorus level is optimal.")

    if current_nutrients['Potassium'] < predicted_nutrients['Applied Potassium (kg/ha)']:
        suggestions.append(f"Increase Potassium by {predicted_nutrients['Applied Potassium (kg/ha)'] - current_nutrients['Potassium']:.2f} kg/ha.")
    elif current_nutrients['Potassium'] > predicted_nutrients['Applied Potassium (kg/ha)']:
        suggestions.append(f"Reduce Potassium by {current_nutrients['Potassium'] - predicted_nutrients['Applied Potassium (kg/ha)']:.2f} kg/ha.")
    else:
        suggestions.append("Potassium level is optimal.")

    return suggestions

@app.route('/')
def index():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM sensor_data ORDER BY date DESC')
    rows = cursor.fetchall()
    conn.close()

    # Prepare chart data
    chart_data = {
        'temperature': [row[2] for row in rows],
        'humidity': [row[3] for row in rows],
        'nitrogen': [row[4] for row in rows],
        'potassium': [row[5] for row in rows],
        'moisture': [row[6] for row in rows],
        'phosphorus': [row[8] for row in rows],
        'soilPH': [row[9] for row in rows],
        'dates': [row[12] for row in rows]
    }
    
    map_data = [
        {
            "lat": row[10],
            "lng": row[11],
            "serial_number": row[1],
            "temperature": row[2],
            "humidity": row[3],
            "nitrogen": row[4],
            "potassium": row[5],
            "moisture": row[6],
            "eclec": row[7],
            "phosphorus": row[8],
            "soilPH": row[9],
            "date": row[12]
        } for row in rows
    ]

    # Fetch all sensor data for charts and maps
    # cursor.execute('SELECT * FROM sensor_data ORDER BY date DESC')
    # rows = cursor.fetchall()
    # conn.close()
    sensor_data = [
            {
                'serial_number': row[1],
                'temperature': row[2],
                'humidity': row[3],
                'nitrogen': row[4],
                'potassium': row[5],
                'moisture': row[6],
                'eclec': row[7],
                'phosphorus': row[8],
                'soilPH': row[9],
                'latitude': row[10],
                'longitude': row[11],
                'date': row[12]
            } for row in rows
        ]
    # Define suggestions_for_all_crops with sample data
    all_crops = ['Potatoes', 'Carrots', 'Beans', 'Tomatoes']
    current_soil_data = {
        'pH': 6.5, 'Nitrogen': 30, 'Phosphorus': 20, 'Potassium': 150  # Sample values
    }
    
    # Generate sample nutrient suggestions for each crop
    suggestions_for_all_crops = {}
    for crop in all_crops:
        suggestions = suggest_soil_nutrients(crop, current_soil_data)
        suggestions_for_all_crops[crop] = suggestions

    return render_template('index.html', chart_data=chart_data, map_data=map_data, suggestions_for_all_crops=suggestions_for_all_crops,sensor_data=sensor_data)

# @app.route('/')
# def index():
#     conn = sqlite3.connect('database.db')
#     cursor = conn.cursor()
#     cursor.execute('SELECT * FROM sensor_data ORDER BY date DESC')
#     rows = cursor.fetchall()

#     # Fetch the latest sensor data
#     cursor.execute('SELECT * FROM sensor_data ORDER BY date DESC LIMIT 1')
#     latest_row = cursor.fetchone()
#     conn.close()

#     if latest_row:
#         # Prepare current soil data for suggestions
#         current_soil_data = {
#             'pH': latest_row[9],  # soilPH
#             'Nitrogen': latest_row[4],
#             'Phosphorus': latest_row[8],
#             'Potassium': latest_row[5]
#         }

#         # Suggest for all crops
#         all_crops = [
#             'Potatoes', 'Carrots', 'Beans', 'Tomatoes', 'Wheat', 'Maize', 'Rice', 'Peppers', 'Cabbage', 'Onions',
#             'Lettuce', 'Spinach', 'Broccoli', 'Peas', 'Strawberries', 'Corn', 'Soybeans', 'Sunflowers', 'Pumpkins',
#             'Sweet Potatoes', 'Grapes', 'Bananas', 'Apples', 'Oranges'
#         ]
        
#         # Generate suggestions for each crop
#         suggestions_for_all_crops = {}
#         for crop in all_crops:
#             suggestions = suggest_soil_nutrients(crop, current_soil_data)
#             suggestions_for_all_crops[crop] = suggestions
#     else:
#         suggestions_for_all_crops = {}

#     if not rows:
#         chart_data = {
#             'temperature': [], 'humidity': [], 'nitrogen': [], 'potassium': [],
#             'moisture': [], 'phosphorus': [], 'soilPH': [], 'dates': []
#         }
#         map_data = []
#     else:
#         chart_data = {
#             'temperature': [row[2] for row in rows],
#             'humidity': [row[3] for row in rows],
#             'nitrogen': [row[4] for row in rows],
#             'potassium': [row[5] for row in rows],
#             'moisture': [row[6] for row in rows],
#             'phosphorus': [row[8] for row in rows],
#             'soilPH': [row[9] for row in rows],
#             'dates': [row[12] for row in rows]
#         }

#         map_data = [
#             {
#                 "lat": row[10], "lng": row[11], "serial_number": row[1],
#                 "temperature": row[2], "humidity": row[3], "nitrogen": row[4],
#                 "potassium": row[5], "moisture": row[6], "eclec": row[7],
#                 "phosphorus": row[8], "soilPH": row[9], "date": row[12]
#             } for row in rows
#         ]

#     return render_template('index.html', suggestions_for_all_crops=suggestions_for_all_crops, sensor_data=rows, chart_data=chart_data, map_data=map_data)

# @app.route('/')
# def index():
#     conn = sqlite3.connect('database.db')
#     cursor = conn.cursor()
#     cursor.execute('SELECT * FROM sensor_data ORDER BY date DESC')
#     rows = cursor.fetchall()

#     # Fetch the latest sensor data
#     cursor.execute('SELECT * FROM sensor_data ORDER BY date DESC LIMIT 1')
#     latest_row = cursor.fetchone()
#     conn.close()

#     if latest_row:
#         # Prepare current soil data for suggestions
#         current_soil_data = {
#             'pH': latest_row[9],  # soilPH
#             'Nitrogen': latest_row[4],
#             'Phosphorus': latest_row[8],
#             'Potassium': latest_row[5]
#         }

#         # Suggest for all crops
#         all_crops = [
#             'Potatoes', 'Carrots', 'Beans', 'Tomatoes', 'Wheat', 'Maize', 'Rice', 'Peppers', 'Cabbage', 'Onions',
#             'Lettuce', 'Spinach', 'Broccoli', 'Peas', 'Strawberries', 'Corn', 'Soybeans', 'Sunflowers', 'Pumpkins',
#             'Sweet Potatoes', 'Grapes', 'Bananas', 'Apples', 'Oranges'
#         ]
        
#         # Generate suggestions for each crop
#         suggestions_for_all_crops = {}
#         for crop in all_crops:
#             suggestions = suggest_soil_nutrients(crop, current_soil_data)
#             suggestions_for_all_crops[crop] = suggestions
#     else:
#         suggestions_for_all_crops = {}

#     if not rows:
#         chart_data = {
#             'temperature': [], 'humidity': [], 'nitrogen': [], 'potassium': [],
#             'moisture': [], 'phosphorus': [], 'soilPH': [], 'dates': []
#         }
#         map_data = []
#     else:
#         chart_data = {
#             'temperature': [row[2] for row in rows],
#             'humidity': [row[3] for row in rows],
#             'nitrogen': [row[4] for row in rows],
#             'potassium': [row[5] for row in rows],
#             'moisture': [row[6] for row in rows],
#             'phosphorus': [row[8] for row in rows],
#             'soilPH': [row[9] for row in rows],
#             'dates': [row[12] for row in rows]
#         }

#         map_data = [
#             {
#                 "lat": row[10], "lng": row[11], "serial_number": row[1],
#                 "temperature": row[2], "humidity": row[3], "nitrogen": row[4],
#                 "potassium": row[5], "moisture": row[6], "eclec": row[7],
#                 "phosphorus": row[8], "soilPH": row[9], "date": row[12]
#             } for row in rows
#         ]

#     return render_template('index.html', suggestions_for_all_crops=suggestions_for_all_crops, sensor_data=rows, chart_data=chart_data, map_data=map_data)

# @app.route('/')
# def index():
#     conn = sqlite3.connect('database.db')
#     cursor = conn.cursor()
#     cursor.execute('SELECT * FROM sensor_data ORDER BY date DESC')
#     rows = cursor.fetchall()

#     # Fetch the latest sensor data
#     cursor.execute('SELECT * FROM sensor_data ORDER BY date DESC LIMIT 1')
#     latest_row = cursor.fetchone()
#     conn.close()

#     if latest_row:
#         # Prepare current soil data for suggestions
#         current_soil_data = {
#             'pH': latest_row[9],  # soilPH
#             'Nitrogen': latest_row[4],
#             'Phosphorus': latest_row[8],
#             'Potassium': latest_row[5]
#         }

#         # List of all crops
#         all_crops = [
#             'Potatoes', 'Carrots', 'Beans', 'Tomatoes'
#         ]
        
#         # Generate suggestions and suitability for each crop
#         crop_suitability = {}
#         suggestions_for_all_crops = {}
#         for crop in all_crops:
#             suggestions = suggest_soil_nutrients(crop, current_soil_data)
#             suggestions_for_all_crops[crop] = suggestions

#             # Calculate the percentage of nutrients that are within range
#             crop_requirements = crop_nutrient_requirements[crop]
#             total_nutrients = len(crop_requirements)
#             in_range_count = 0
#             reasons = []

#             for nutrient, (min_value, max_value) in crop_requirements.items():
#                 current_value = current_soil_data.get(nutrient)
#                 if current_value is None:
#                     reasons.append(f"Missing {nutrient} data.")
#                 elif min_value <= current_value <= max_value:
#                     in_range_count += 1  # Nutrient is within range
#                 else:
#                     if current_value < min_value:
#                         reasons.append(f"Low {nutrient}: Current {current_value}, Minimum {min_value}.")
#                     elif current_value > max_value:
#                         reasons.append(f"High {nutrient}: Current {current_value}, Maximum {max_value}.")

#             # Calculate percentage suitability
#             suitability_percentage = (in_range_count / total_nutrients) * 100
#             crop_suitability[crop] = {
#                 'percentage': suitability_percentage,
#                 'reasons': reasons
#             }

#         # Find the crop with the highest suitability percentage
#         best_crop = max(crop_suitability, key=lambda x: crop_suitability[x]['percentage'])
#     else:
#         suggestions_for_all_crops = {}
#         best_crop = None
#         crop_suitability = {}

#     # Prepare chart data and map data as before
#     if not rows:
#         chart_data = {
#             'temperature': [], 'humidity': [], 'nitrogen': [], 'potassium': [],
#             'moisture': [], 'phosphorus': [], 'soilPH': [], 'dates': []
#         }
#         map_data = []
#     else:
#         chart_data = {
#             'temperature': [row[2] for row in rows],
#             'humidity': [row[3] for row in rows],
#             'nitrogen': [row[4] for row in rows],
#             'potassium': [row[5] for row in rows],
#             'moisture': [row[6] for row in rows],
#             'elec': [row[7] for row in rows],
#             'phosphorus': [row[8] for row in rows],
#             'soilPH': [row[9] for row in rows],
#             'dates': [row[12] for row in rows]
#         }

#         map_data = [
#             {
#                 "lat": row[10], "lng": row[11], "serial_number": row[1],
#                 "temperature": row[2], "humidity": row[3], "nitrogen": row[4],
#                 "potassium": row[5], "moisture": row[6], "elec": row[7],
#                 "phosphorus": row[8], "soilPH": row[9], "date": row[12]
#             } for row in rows
#         ]

#     return render_template('index.html', suggestions_for_all_crops=suggestions_for_all_crops, best_crop=best_crop, crop_suitability=crop_suitability, sensor_data=rows, chart_data=chart_data, map_data=map_data)

# Dictionary to map crop names to models
models = {
    'potatoes': potato_model,
    'carrots': carrot_model,
    'beans': bean_model,
    'tomatoes': tomato_model
}

# Function to make predictions for a given crop and steps
def make_prediction(model, crop, steps):
    # Use the ARIMA model to predict for the next 'steps' months
    forecast = model.forecast(steps=steps).tolist()
    
    # Add 430 to bean predictions
    if crop == 'beans':
        forecast = [price + 430 for price in forecast]
    
    return forecast

# # API route to get predictions for multiple crops
# @app.route('/predict', methods=['GET'])
# def predict():
#     # Get multiple crops and steps (number of months to predict)
#     crops = request.args.get('crops', '').lower().split(',')  # crops is now a comma-separated list
#     steps = int(request.args.get('steps', 12))  # Default to 12 months
    
#     # Validate crop inputs and ensure all are valid
#     invalid_crops = [crop for crop in crops if crop not in models]
#     if invalid_crops:
#         return jsonify({'error': f"Invalid crops: {', '.join(invalid_crops)}. Choose from potatoes, carrots, beans, tomatoes"}), 400
    
#     # Make predictions for all valid crops
#     predictions = {}
#     for crop in crops:
#         model = models[crop]
#         predictions[crop] = make_prediction(model, crop, steps)
    
#     # Return predictions for all selected crops
#     return jsonify(predictions)

# @app.route('/suggest_nutrients', methods=['POST'])
# def suggest_nutrients():
#     # Get selected crop from form
#     selected_crop = request.form.get('crop')

#     # Retrieve the latest sensor data
#     conn = sqlite3.connect('database.db')
#     cursor = conn.cursor()
#     cursor.execute('SELECT * FROM sensor_data ORDER BY date DESC LIMIT 1')
#     latest_row = cursor.fetchone()

#     if latest_row:
#         current_soil_data = {
#             'pH': latest_row[9],  # soilPH
#             'Nitrogen': latest_row[4],
#             'Phosphorus': latest_row[8],
#             'Potassium': latest_row[5]
#         }
#     else:
#         # If no sensor data is available, return default or error
#         current_soil_data = {
#             'pH': None,
#             'Nitrogen': None,
#             'Phosphorus': None,
#             'Potassium': None
#         }

#     # Generate suggestions for the selected crop
#     suggestions = suggest_soil_nutrients(selected_crop, current_soil_data)
# @app.route('/suggest_nutrients', methods=['POST'])
# def suggest_nutrients():
#     # Get selected crop from the form
#     selected_crop = request.form.get('crop').lower()  # convert crop name to lowercase to match model keys

#     # Retrieve the latest sensor data
#     conn = sqlite3.connect('database.db')
#     cursor = conn.cursor()
#     cursor.execute('SELECT * FROM sensor_data ORDER BY date DESC LIMIT 1')
#     latest_row = cursor.fetchone()

#     if latest_row:
#         current_soil_data = {
#             'pH': latest_row[9],  # soilPH
#             'Nitrogen': latest_row[4],
#             'Phosphorus': latest_row[8],
#             'Potassium': latest_row[5]
#         }
#     else:
#         # If no sensor data is available, return default or error
#         current_soil_data = {
#             'pH': None,
#             'Nitrogen': None,
#             'Phosphorus': None,
#             'Potassium': None
#         }

#     # Generate suggestions for the selected crop
#     suggestions = suggest_soil_nutrients(selected_crop.capitalize(), current_soil_data)

#     # Make a prediction for the selected crop's price using the predict function
#     if selected_crop in models:
#         steps = 12  # default to predicting 12 months ahead
#         prediction_model = models[selected_crop]
#         price_predictions = make_prediction(prediction_model, selected_crop, steps)
#     else:
#         price_predictions = []

#     return render_template('index.html', suggestions=suggestions, selected_crop=selected_crop.capitalize(), price_predictions=price_predictions)
#     # Fetch all sensor data to render in the charts
#     cursor.execute('SELECT * FROM sensor_data ORDER BY date DESC')
#     rows = cursor.fetchall()
#     conn.close()

#     # Prepare data for charts and map
#     chart_data = {
#         'temperature': [row[2] for row in rows],
#         'humidity': [row[3] for row in rows],
#         'nitrogen': [row[4] for row in rows],
#         'potassium': [row[5] for row in rows],
#         'moisture': [row[6] for row in rows],
#         'phosphorus': [row[8] for row in rows],
#         'soilPH': [row[9] for row in rows],
#         'elec': [row[7] for row in rows],
#         'dates': [row[12] for row in rows]
#     }

#     map_data = [
#         {
#             "lat": row[10],
#             "lng": row[11],
#             "serial_number": row[1],
#             "temperature": row[2],
#             "humidity": row[3],
#             "nitrogen": row[4],
#             "potassium": row[5],
#             "moisture": row[6],
#             "eclec": row[7],
#             "phosphorus": row[8],
#             "soilPH": row[9],
#             "date": row[12]
#         } for row in rows
#     ]

#     return render_template('index.html', suggestions=suggestions, selected_crop=selected_crop, price_predictions=price_predictions, chart_data=chart_data, map_data=map_data)

@app.route('/suggest_nutrients', methods=['POST'])
def suggest_nutrients():
    # Get selected crop from form
    selected_crop = request.form.get('crop').capitalize()

    # Retrieve the latest sensor data
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM sensor_data ORDER BY date DESC LIMIT 1')
    latest_row = cursor.fetchone()
    
    # Prepare current soil data if sensor data is available
    if latest_row:
        current_soil_data = {
            'pH': latest_row[9],  # soilPH
            'Nitrogen': latest_row[4],
            'Phosphorus': latest_row[8],
            'Potassium': latest_row[5]
        }
    else:
        current_soil_data = {
            'pH': None,
            'Nitrogen': None,
            'Phosphorus': None,
            'Potassium': None
        }

    # Generate nutrient suggestions
    suggestions = suggest_soil_nutrients(selected_crop, current_soil_data)

    # Make predictions if the crop model exists
    if selected_crop.lower() in models:
        steps = 12  # default to predicting 12 months ahead
        prediction_model = models[selected_crop.lower()]
        price_predictions = make_prediction(prediction_model, selected_crop.lower(), steps)
    else:
        price_predictions = []

    # Fetch all sensor data for charts and maps
    cursor.execute('SELECT * FROM sensor_data ORDER BY date DESC')
    rows = cursor.fetchall()
    conn.close()
    sensor_data = [
            {
                'serial_number': row[1],
                'temperature': row[2],
                'humidity': row[3],
                'nitrogen': row[4],
                'potassium': row[5],
                'moisture': row[6],
                'eclec': row[7],
                'phosphorus': row[8],
                'soilPH': row[9],
                'latitude': row[10],
                'longitude': row[11],
                'date': row[12]
            } for row in rows
        ]
    # Prepare chart data for temperature, humidity, etc.
    chart_data = {
        'temperature': [row[2] for row in rows],
        'humidity': [row[3] for row in rows],
        'nitrogen': [row[4] for row in rows],
        'potassium': [row[5] for row in rows],
        'moisture': [row[6] for row in rows],
        'phosphorus': [row[8] for row in rows],
        'soilPH': [row[9] for row in rows],
        'dates': [row[12] for row in rows]
    }

    # Prepare map data for sensor locations
    map_data = [
        {
            "lat": row[10],
            "lng": row[11],
            "serial_number": row[1],
            "temperature": row[2],
            "humidity": row[3],
            "nitrogen": row[4],
            "potassium": row[5],
            "moisture": row[6],
            "eclec": row[7],
            "phosphorus": row[8],
            "soilPH": row[9],
            "date": row[12]
        } for row in rows
    ]

    return render_template(
        'index.html',
        suggestions=suggestions,
        selected_crop=selected_crop,
        price_predictions=price_predictions,
        chart_data=chart_data,
        map_data=map_data,
        sensor_data=sensor_data
    )


if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
