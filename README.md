# Sign Language Recognition System

This project implements a sign language recognition system that can identify seven different signs. The system uses MediaPipe landmarks and a TensorFlow model to recognize hand gestures in real time.

## Project Structure

- `isl1.ipynb`: Jupyter notebook containing the model training code
- `action_best.h5`: Trained model file used by the app
- `app/`: Web application for real-time sign language recognition
  - `app.py`: Flask application
  - `templates/`: HTML templates
  - `static/`: CSS and JavaScript files
  - `requirements.txt`: Dependencies

## Web Application

The web application provides a user-friendly interface for real-time sign language recognition. It captures webcam frames in the browser, extracts landmarks with MediaPipe JS, sends sequence keypoints to a Flask prediction endpoint, and displays recognized signs as text.

### Features

- Real-time sign language recognition
- Visual feedback with landmarks on hands, face, and pose
- Display of recognized signs as text
- User-friendly interface with instructions

## Model Training

The model was trained using the following steps:

1. Data collection: Recording videos of different sign language gestures
2. Feature extraction: Using MediaPipe to extract keypoints from hands, face, and pose
3. Model training: Using a LSTM neural network to recognize patterns in the keypoint sequences
4. Model evaluation: Testing the model on unseen data

## Getting Started

To run the web application, follow these steps:

1. Navigate to the app directory:

   ```
   cd app
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Start the Flask application:

   ```
   python app.py
   ```

4. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5002/
   ```

## Free Deployment (Student Friendly)

This repo now supports split deployment:

- Backend (model inference API): Render free tier
- Frontend (camera + MediaPipe UI): Vercel free tier

### 1) Deploy backend on Render

1. Push this repository to GitHub.
2. In Render, create a new Blueprint/Web Service from the repo.
3. Render will detect `render.yaml` and deploy `app/` as the backend service.
4. After deploy, open `https://<your-render-service>.onrender.com/health` and confirm status is ok.

### 2) Deploy frontend on Vercel

1. Import the same GitHub repository in Vercel.
2. Set **Root Directory** to `app`.
3. Deploy.
4. Edit `app/index.html` and set `window.APP_CONFIG.apiBaseUrl` to your Render backend URL.
5. Redeploy Vercel after updating the backend URL.

### 3) Production check

1. Open your Vercel URL in browser.
2. Click **Start Recognition** and allow webcam permission.
3. Wait for backend wakeup (free-tier cold start), then verify live predictions.

For more detailed instructions, see the README.md file in the app directory.

## License

This project is licensed under the MIT License.
