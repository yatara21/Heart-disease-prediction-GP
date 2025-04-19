from flask import Flask, request, render_template, flash, redirect, url_for
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Simplified secret key

# Load model and features
model = joblib.load("models/heart_disease_rf_model.pkl")
feature_order = joblib.load("models/model_features.pkl")

def validate_numeric_input(value, field_name, min_val, max_val):
    try:
        value = float(value)
        if not (min_val <= value <= max_val):
            raise ValueError(f"{field_name} must be between {min_val} and {max_val}")
        return value
    except ValueError:
        raise ValueError(f"{field_name} must be a valid number")

def create_feature_vector(form_data):
    # Create input data
    input_data = {
        'Age': validate_numeric_input(form_data['Age'], 'Age', 28, 77),
        'RestingBP': validate_numeric_input(form_data['RestingBP'], 'RestingBP', 80, 200),
        'Cholesterol': validate_numeric_input(form_data['Cholesterol'], 'Cholesterol', 85, 603),
        'FastingBS': int(form_data['FastingBS']),
        'MaxHR': validate_numeric_input(form_data['MaxHR'], 'MaxHR', 60, 202),
        'Oldpeak': validate_numeric_input(form_data['Oldpeak'], 'Oldpeak', -2.6, 6.2),
        'Sex_M': int(form_data['Sex_M']),
        'ExerciseAngina_Y': int(form_data['ExerciseAngina_Y'])
    }
    
    print("\n=== Processing new prediction request ===")
    print(f"Input data: {form_data}")
    
    # Create DataFrame with all possible features initialized to 0
    df = pd.DataFrame(columns=feature_order, data=np.zeros((1, len(feature_order))))
    
    # Update numeric and binary features
    for feature, value in input_data.items():
        if feature in feature_order:
            df[feature] = value
            print(f"Set {feature} = {value}")
    
    # Handle categorical features
    for feature, value in {'ChestPainType': form_data['ChestPainType'],
                          'RestingECG': form_data['RestingECG'],
                          'ST_Slope': form_data['ST_Slope']}.items():
        feature_name = f"{feature}_{value}"
        if feature_name in feature_order:
            df[feature_name] = 1
            print(f"Set {feature_name} = 1")
    
    print("\nFinal feature vector:")
    for col in df.columns:
        if df[col].values[0] != 0:
            print(f"{col}: {df[col].values[0]}")
    
    return df

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            input_df = create_feature_vector(request.form)
            pred = model.predict(input_df)[0]
            pred_proba = model.predict_proba(input_df)[0]
            
            print(f"Probabilities: No: {pred_proba[0]:.2%}, Yes: {pred_proba[1]:.2%}")
            
            result = "Positive" if pred == 1 else "Negative"
            confidence = max(pred_proba)
            message = f"Heart Disease Prediction: {result}\nConfidence: {confidence:.2%}"
            
            flash(message, 'success' if confidence > 0.7 else 'warning')
            return redirect(url_for('predict'))
            
        except ValueError as e:
            flash(str(e), 'error')
            return redirect(url_for('predict'))
        except Exception as e:
            flash("An unexpected error occurred.", 'error')
            return redirect(url_for('predict'))

    return render_template("form.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)