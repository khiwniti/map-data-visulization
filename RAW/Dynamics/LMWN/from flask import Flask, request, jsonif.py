from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Save the model
joblib.dump(best_model, 'model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

# Load the model
model = joblib.load('model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    X = preprocessor.transform(df)
    prediction = model.predict(X)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)