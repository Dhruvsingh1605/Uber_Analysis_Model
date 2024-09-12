import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # Ensure the request has JSON content type
    if request.content_type != 'application/json':
        return jsonify({'error': 'Invalid content-type, must be application/json'}), 400

    # Retrieve the data from the request
    data = request.json.get('data')
    print(f"Received data: {data}")

    # Check if data is provided
    if data is None:
        return jsonify({'error': 'No data provided'}), 400

    try:
        # Convert data to a NumPy array for model prediction
        new_data = np.array(list(data.values())).reshape(1, -1)
    except Exception as e:
        print(f"Error processing data: {e}")
        return jsonify({'error': 'Error processing data'}), 500

    # Make the prediction
    output = regmodel.predict(new_data)
    print(f'Predicted fare: {output[0]}')

    # Return the prediction as a JSON response
    return jsonify({'prediction': f'The predicted fare is ${output[0]:.2f}'})

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    miles = request.form['miles']
    category = request.form['category']

    # Convert form data to NumPy array for prediction
    final_input = np.array([[float(miles), int(category)]])
    print(f'Final input: {final_input}')

    # Make the prediction
    output = regmodel.predict(final_input)
    print(f'Predicted fare: {output[0]}')

    # Return the result and render on home page
    return render_template("home.html", prediction_text=f'The Fare is: ${output[0]:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
