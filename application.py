from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = tf.keras.models.load_model("cnn_rnn_model.h5")
scaler = joblib.load('scaler.pkl')  # load saved scaler from training phase

@app.route('/')
def index():
    return render_template("form.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs
        heart_rate = float(request.form['heart_rate'])
        blood_pressure = float(request.form['blood_pressure'])
        oxygen_saturation = float(request.form['oxygen_saturation'])
        respiratory_rate = float(request.form['respiratory_rate'])
        temperature = float(request.form['temperature'])

        # Clip inputs
        heart_rate = np.clip(heart_rate, 40, 180)
        blood_pressure = np.clip(blood_pressure, 60, 200)
        oxygen_saturation = np.clip(oxygen_saturation, 80, 100)
        respiratory_rate = np.clip(respiratory_rate, 10, 40)
        temperature = np.clip(temperature, 30, 43)

        # Form array
        input_data = np.array([[heart_rate, blood_pressure, oxygen_saturation, respiratory_rate, temperature]])

        # Scale and reshape
        input_scaled = scaler.transform(input_data)
        input_reshaped = input_scaled.reshape(1, input_scaled.shape[1], 1)

        # Predict
        prediction = model.predict(input_reshaped)
        label = "Anomaly" if prediction[0][0] > 0.5 else "Normal"

        return render_template("result_realtime.html", prediction=label, probability=float(prediction[0][0]))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
