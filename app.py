from flask import Flask, request, render_template, flash, redirect, url_for
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)


# Load model and features
try:
    model = joblib.load("heart_disease_rf_model(1).pkl")
    feature_order = joblib.load("model_features.pkl")
except Exception as e:
    print(f"Error loading model files: {e}")
    raise

def validate_numeric_input(value, field_name, min_val, max_val):
    try:
        value = float(value)
        if not (min_val <= value <= max_val):
            raise ValueError
        return value
    except ValueError:
        raise ValueError(f"{field_name} must be between {min_val} and {max_val}")

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            form = request.form
            
            # Validate numerical features with proper bounds
            data = {
                'Age': validate_numeric_input(form['Age'], 'Age', 28, 77),
                'RestingBP': validate_numeric_input(form['RestingBP'], 'Resting Blood Pressure', 80, 200),
                'Cholesterol': validate_numeric_input(form['Cholesterol'], 'Cholesterol', 85, 603),
                'FastingBS': int(form['FastingBS']),
                'MaxHR': validate_numeric_input(form['MaxHR'], 'Max Heart Rate', 60, 202),
                'Oldpeak': validate_numeric_input(form['Oldpeak'], 'ST Depression', -2.6, 6.2),
            }

            # Initialize all categorical features to 0
            for col in feature_order:
                if col not in data:
                    data[col] = 0

            # Handle categorical features
            data['Sex_M'] = int(form['Sex_M'])
            data['ExerciseAngina_Y'] = int(form['ExerciseAngina_Y'])

            # Set one-hot encoded values
            chest_pain = f"ChestPainType_{form['ChestPainType']}"
            if chest_pain in feature_order:
                data[chest_pain] = 1

            ecg = f"RestingECG_{form['RestingECG']}"
            if ecg in feature_order:
                data[ecg] = 1

            slope = f"ST_Slope_{form['ST_Slope']}"
            if slope in feature_order:
                data[slope] = 1

            # Make prediction
            input_df = pd.DataFrame([data])[feature_order]
            pred = model.predict(input_df)[0]
            pred_proba = model.predict_proba(input_df)[0]
            
            confidence = pred_proba[1] if pred == 1 else pred_proba[0]
            result = "Positive" if pred == 1 else "Negative"
            message = f"{result} for Heart Disease (Confidence: {confidence:.2%})"

            # After prediction, redirect to show results
            flash(message, 'success')
            return redirect(url_for('predict'))

        except ValueError as e:
            flash(str(e), 'error')
            return redirect(url_for('predict'))
        except Exception as e:
            flash("An error occurred while processing your request.", 'error')
            return redirect(url_for('predict'))

    # GET request - show empty form
    return render_template("form.html")

if __name__ == "__main__":
    app.run(debug=True)
