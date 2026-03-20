# Sign Language Recognition Web Application

This is a web application for real-time sign language recognition using a pre-trained model. The application uses computer vision and deep learning to recognize signs and translate them to text.

## Features

- Browser-based webcam capture (camera runs on client device)
- MediaPipe JS landmark extraction in real time
- Browser TensorFlow.js inference (primary path)
- Backend `/predict` fallback (optional)
- Visual feedback with face, pose, and hand landmarks
- Stable recognized sentence with confidence display

## Technologies Used

- Flask: Web framework for serving the application
- TensorFlow: Deep learning framework for running the sign language recognition model
- MediaPipe JS: Browser framework for landmark detection
- HTML/CSS/JavaScript: Frontend technologies for camera + UI

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Webcam access in browser

### Installation

1. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Make sure the model file (`action_best.h5`) is located in the parent directory.

### Running the Application

1. Start the Flask application:

   ```
   python app.py
   ```

2. Open your web browser and navigate to:

   ```
   http://127.0.0.1:5002/
   ```

3. Click "Start Recognition" and allow webcam permission.

### Model conversion for browser inference

1. Install converter dependencies (optional, may require a separate Python env if dependency conflicts appear):

   ```
   pip install tensorflowjs
   ```

2. Convert the model:

   ```
   python convert_to_tfjs.py
   ```

3. Expected output location:

   ```
   static/models/action_best_tfjs/model.json
   ```

If conversion fails, keep `useBackendFallback: true` in frontend config so predictions use `/predict`.

### Split Deployment (Vercel frontend + separate backend)

- Keep this Flask app deployed as backend (for `/predict`).
- Deploy frontend from the `app/` directory on Vercel.
- Use `app/index.html` and `app/vercel.json` for static deployment.
- In `app/index.html`, set `window.APP_CONFIG.apiBaseUrl` to your backend URL.
- Example: `https://your-backend.onrender.com`
- Keep `useBackendFallback: true` unless browser model conversion succeeds.

### Render backend (quick setup)

Use these values if configuring manually instead of Blueprint:

- Root Directory: `app`
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
- Health Check Path: `/health`

## Usage

1. Position yourself in a well-lit area with the camera facing you.
2. Click the "Start Recognition" button.
3. Allow browser camera permission when prompted.
4. After sequence warmup, begin signing one of the available signs.
5. Hold each sign steady for better recognition stability.
6. The recognized signs and confidence appear in the panel.
7. Use "Stop Recognition" to pause or "Reset" to clear sequence history.

## Available Signs

The model can recognize the following signs:

- cold
- fever
- cough
- medication
- injection
- operation
- pain

## API Contract

### `POST /predict`

Request body:

- `sequence`: array with shape `(30, 1662)` of keypoint vectors

Response body:

- `action`: predicted sign label
- `confidence`: score for predicted label
- `probabilities`: full per-class probabilities
- `actions`: class labels order

## License

This project is licensed under the MIT License.
