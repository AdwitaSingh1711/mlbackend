# pylint: disable=pointless-string-statement
from flask import Flask, jsonify, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('crop_recommendation.pkl')

# define endpoint for the prediction API
@app.route('/', methods=['POST'])
def predict():
    # get the parameters from the request
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['k'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    pH = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    
    # create a numpy array from the input parameters
    input_data = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
    
    # make a prediction using the loaded model
    prediction = model.predict(input_data)
    
    # return the prediction as a JSON object
    return jsonify({'crop_label': prediction[0].item()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=81)

    '''
    @app.route('/price_prediction', methods=['POST'])
    def price():
    Month = float(request.form['Month'])
    Year = float(request.form['Year'])
    rainfall_new = float(request.form['rainfall_new'])
    

    '''