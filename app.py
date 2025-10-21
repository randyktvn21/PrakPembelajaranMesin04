from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import math
import os

app = Flask(__name__)

# Load model dan scaler
model = load_model('model_lstm_suhu.h5')
scaler = joblib.load('scaler_suhu.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil input dari form
        data = request.form.get('sequence')
        sequence = [float(x) for x in data.split(',')]

        # Pastikan panjang data = 60
        if len(sequence) != 60:
            return render_template('index.html', 
                                   error="Harap masukkan tepat 60 angka suhu (misalnya dari 15.0 hingga 30.0).")

        # Scaling dan reshape
        scaled_seq = scaler.transform(np.array(sequence).reshape(-1, 1))
        X_input = np.array([scaled_seq])
        pred_scaled = model.predict(X_input)
        pred_temp = scaler.inverse_transform(pred_scaled)[0][0]

        # Dummy nilai evaluasi (misal sudah dihitung saat training)
        rmse = 1.25
        mae = 0.89

        return render_template('index.html',
                               prediction=f"{pred_temp:.2f} °C",
                               rmse=rmse,
                               mae=mae)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
