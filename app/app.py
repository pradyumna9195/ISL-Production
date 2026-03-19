import os
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Load the model
model_path = Path(__file__).resolve().parent.parent / 'action_best.h5'
model = tf.keras.models.load_model(model_path)

# Actions/signs that the model can recognize
actions = np.array(['cold', 'fever', 'cough', 'medication', 'injection', 'operation', 'pain'])
sequence_length = 30
feature_length = 1662

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model_loaded": True})


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(silent=True)
    if not payload or 'sequence' not in payload:
        return jsonify({"error": "Missing 'sequence' in JSON body"}), 400

    try:
        sequence = np.array(payload['sequence'], dtype=np.float32)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid 'sequence' format"}), 400

    if sequence.shape != (sequence_length, feature_length):
        return jsonify({
            "error": "Invalid sequence shape",
            "expected": [sequence_length, feature_length],
            "received": list(sequence.shape)
        }), 400

    prediction = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    confidence = float(prediction[predicted_index])

    return jsonify({
        "action": actions[predicted_index],
        "confidence": confidence,
        "probabilities": prediction.tolist(),
        "actions": actions.tolist()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)