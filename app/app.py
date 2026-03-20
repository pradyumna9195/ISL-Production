import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, jsonify, render_template, Response, request
from pathlib import Path

app = Flask(__name__)

# Load the model
model_path = Path(__file__).resolve().parent.parent / 'action_best.h5'
model = tf.keras.models.load_model(model_path)

# Actions/signs that the model can recognize
actions = np.array(['cold', 'fever', 'cough', 'medication', 'injection', 'operation', 'pain'])
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 0)]
sequence_length = 30
feature_length = 1662

# Initialize MediaPipe components
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model_instance):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model_instance.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
    )

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
    )

    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )

    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])


def prob_viz(res, action_list, input_frame, color_list):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), color_list[num % len(color_list)], -1)
        cv2.putText(output_frame, action_list[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame


sequence = []
sentence = []
predictions = []
threshold = 0.4
current_status = "Waiting for signs..."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_status', methods=['GET'])
def get_status():
    global current_status
    return jsonify({"status": current_status})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model_loaded": True})


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(silent=True)
    if not payload or 'sequence' not in payload:
        return jsonify({"error": "Missing 'sequence' in JSON body"}), 400

    try:
        sequence_data = np.array(payload['sequence'], dtype=np.float32)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid 'sequence' format"}), 400

    if sequence_data.shape != (sequence_length, feature_length):
        return jsonify({
            "error": "Invalid sequence shape",
            "expected": [sequence_length, feature_length],
            "received": list(sequence_data.shape)
        }), 400

    prediction = model.predict(np.expand_dims(sequence_data, axis=0), verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    confidence = float(prediction[predicted_index])

    return jsonify({
        "action": str(actions[predicted_index]),
        "confidence": confidence,
        "probabilities": prediction.tolist(),
        "actions": actions.tolist()
    })


def generate_frames():
    global sequence, sentence, predictions, current_status

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        cap.set(3, 640)
        cap.set(4, 480)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture image")
                    break

                try:
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-sequence_length:]

                    if len(sequence) == sequence_length:
                        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                        predictions.append(np.argmax(res))

                        image = prob_viz(res, actions, image, colors)

                        if len(predictions) >= 10 and np.unique(predictions[-10:])[0] == np.argmax(res):
                            if res[np.argmax(res)] > threshold:
                                if len(sentence) > 0:
                                    if actions[np.argmax(res)] != sentence[-1]:
                                        sentence.append(actions[np.argmax(res)])
                                else:
                                    sentence.append(actions[np.argmax(res)])

                        if len(sentence) > 5:
                            sentence = sentence[-5:]

                        current_status = ' '.join(sentence) if sentence else "Waiting for signs..."

                    cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                    cv2.putText(
                        image,
                        ' '.join(sentence),
                        (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                    flag, buffer = cv2.imencode('.jpg', image)
                    if not flag:
                        continue

                    frame_bytes = buffer.tobytes()

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                except Exception as exc:
                    print(f"Error in frame processing: {exc}")
                    continue
        finally:
            cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)