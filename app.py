from flask import Flask,render_template,request,jsonify
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('hand_writen_detector_log_cls.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    grid_data = np.array(data['data'])

    # Predict the digit using the logistic regression model
    prediction = model.predict([grid_data])[0]
    
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)