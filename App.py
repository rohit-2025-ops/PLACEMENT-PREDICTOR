from flask import Flask, render_template, request
import pickle
import numpy as np

# Load model
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs
        cgpa = request.form['cgpa']
        iq = request.form['iq']

        # Validate numeric input
        cgpa_val = float(cgpa)
        iq_val = float(iq)

        # Predict
        features = np.array([[cgpa_val, iq_val]])
        prediction = model.predict(features)[0]

        if prediction == 1:
            result = '🎓 Placement: Likely to be Placed ✅'
        else:
            result = '🎓 Placement: Not Likely to be Placed ❌'

        return render_template('index.html', prediction_text=result)

    except ValueError:
        # Handles text or invalid numeric input
        return render_template('index.html', prediction_text='⚠️ Please enter valid numeric values for CGPA and IQ.')

if __name__ == "__main__":
    app.run(debug=True)
