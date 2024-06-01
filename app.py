from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('house_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data['MedInc'], data['HouseAge'], data['AveRooms'], data['AveBedrms'], data['Population'], data['AveOccup'], data['Latitude'], data['Longitude']]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
