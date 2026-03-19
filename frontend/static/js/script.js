document.addEventListener("DOMContentLoaded", () => {
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");
  const resetBtn = document.getElementById("resetBtn");
  const videoElement = document.getElementById("input-video");
  const canvasElement = document.getElementById("output-canvas");
  const canvasCtx = canvasElement.getContext("2d");
  const sentenceElement = document.getElementById("sentence");
  const statusElement = document.getElementById("status");
  const confidenceElement = document.getElementById("confidence");
  const cameraStateElement = document.getElementById("cameraState");

  const actions = [
    "cold",
    "fever",
    "cough",
    "medication",
    "injection",
    "operation",
    "pain",
  ];
  const threshold = 0.4;
  const sequenceLength = 30;
  const stablePredictionCount = 10;
  const apiBaseUrl =
    window.APP_CONFIG && window.APP_CONFIG.apiBaseUrl
      ? window.APP_CONFIG.apiBaseUrl.replace(/\/$/, "")
      : "";

  let sequence = [];
  let predictions = [];
  let sentence = [];
  let isRunning = false;
  let isPredicting = false;
  let holistic = null;
  let camera = null;
  let frameCounter = 0;

  stopBtn.disabled = true;
  resetBtn.disabled = true;
  statusElement.textContent = "Waiting for signs...";

  startBtn.addEventListener("click", startRecognition);
  stopBtn.addEventListener("click", stopRecognition);
  resetBtn.addEventListener("click", resetRecognition);

  async function warmupBackend() {
    if (!apiBaseUrl) {
      return true;
    }

    statusElement.textContent = "Waking backend service...";

    try {
      const response = await fetch(`${apiBaseUrl}/health`, { method: "GET" });
      if (!response.ok) {
        throw new Error(`Health check failed with status ${response.status}`);
      }
      return true;
    } catch (error) {
      console.error("Health check error:", error);
      statusElement.textContent = "Backend unavailable. Please retry in a few seconds.";
      return false;
    }
  }

  async function setupHolistic() {
    if (holistic) {
      return;
    }

    holistic = new Holistic({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
    });

    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    holistic.onResults(onResults);
  }

  async function startRecognition() {
    if (isRunning) {
      return;
    }

    startBtn.disabled = true;

    const backendReady = await warmupBackend();
    if (!backendReady) {
      startBtn.disabled = false;
      return;
    }

    await setupHolistic();

    try {
      camera = new Camera(videoElement, {
        onFrame: async () => {
          if (isRunning && holistic) {
            await holistic.send({ image: videoElement });
          }
        },
        width: 640,
        height: 480,
      });

      await camera.start();

      isRunning = true;
      stopBtn.disabled = false;
      resetBtn.disabled = false;
      cameraStateElement.textContent = "Camera is live";
      statusElement.textContent = "Collecting landmarks...";
    } catch (error) {
      console.error("Failed to start camera:", error);
      cameraStateElement.textContent =
        "Camera permission denied or unavailable";
      statusElement.textContent = "Please allow camera access and try again.";
      startBtn.disabled = false;
    }
  }

  function stopRecognition() {
    if (!isRunning) {
      return;
    }

    isRunning = false;
    if (camera) {
      camera.stop();
      camera = null;
    }

    const stream = videoElement.srcObject;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      videoElement.srcObject = null;
    }

    startBtn.disabled = false;
    stopBtn.disabled = true;
    resetBtn.disabled = false;
    cameraStateElement.textContent = "Camera is off";
    statusElement.textContent = "Recognition paused";
  }

  function resetRecognition() {
    sequence = [];
    predictions = [];
    sentence = [];
    frameCounter = 0;
    isPredicting = false;

    sentenceElement.textContent = "";
    statusElement.textContent = isRunning
      ? "Collecting landmarks..."
      : "Waiting for signs...";
    confidenceElement.textContent = "Confidence: --";
  }

  function onResults(results) {
    const width = results.image.width;
    const height = results.image.height;

    canvasElement.width = width;
    canvasElement.height = height;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, width, height);
    canvasCtx.drawImage(results.image, 0, 0, width, height);

    if (results.faceLandmarks) {
      drawConnectors(canvasCtx, results.faceLandmarks, FACEMESH_TESSELATION, {
        color: "#C0C0C070",
        lineWidth: 1,
      });
    }
    if (results.poseLandmarks) {
      drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {
        color: "#00FF00",
        lineWidth: 2,
      });
      drawLandmarks(canvasCtx, results.poseLandmarks, {
        color: "#FF0000",
        lineWidth: 1,
      });
    }
    if (results.leftHandLandmarks) {
      drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, {
        color: "#CC0000",
        lineWidth: 3,
      });
      drawLandmarks(canvasCtx, results.leftHandLandmarks, {
        color: "#00FF00",
        lineWidth: 1,
      });
    }
    if (results.rightHandLandmarks) {
      drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, {
        color: "#00CC00",
        lineWidth: 3,
      });
      drawLandmarks(canvasCtx, results.rightHandLandmarks, {
        color: "#FF0000",
        lineWidth: 1,
      });
    }

    canvasCtx.restore();

    const keypoints = extractKeypoints(results);
    sequence.push(keypoints);
    sequence = sequence.slice(-sequenceLength);

    if (sequence.length < sequenceLength) {
      statusElement.textContent = `Collecting landmarks... ${sequence.length}/${sequenceLength}`;
      return;
    }

    frameCounter += 1;
    if (frameCounter % 2 !== 0 || isPredicting) {
      return;
    }

    sendForPrediction(sequence);
  }

  function extractKeypoints(results) {
    const pose = flattenLandmarks(results.poseLandmarks, 33, true);
    const face = flattenLandmarks(results.faceLandmarks, 468, false);
    const leftHand = flattenLandmarks(results.leftHandLandmarks, 21, false);
    const rightHand = flattenLandmarks(results.rightHandLandmarks, 21, false);
    return [...pose, ...face, ...leftHand, ...rightHand];
  }

  function flattenLandmarks(landmarks, expectedCount, includeVisibility) {
    const dimensions = includeVisibility ? 4 : 3;
    if (!landmarks || landmarks.length !== expectedCount) {
      return new Array(expectedCount * dimensions).fill(0);
    }

    const flattened = [];
    for (let i = 0; i < landmarks.length; i += 1) {
      const landmark = landmarks[i];
      flattened.push(landmark.x, landmark.y, landmark.z);
      if (includeVisibility) {
        flattened.push(
          typeof landmark.visibility === "number" ? landmark.visibility : 0,
        );
      }
    }

    return flattened;
  }

  async function sendForPrediction(currentSequence) {
    isPredicting = true;

    try {
      const response = await fetch(`${apiBaseUrl}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ sequence: currentSequence }),
      });

      if (!response.ok) {
        throw new Error(`Prediction failed with status ${response.status}`);
      }

      const data = await response.json();
      const actionIndex = actions.indexOf(data.action);

      if (actionIndex === -1) {
        return;
      }

      const confidence = Number(data.confidence || 0);
      confidenceElement.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;

      predictions.push(actionIndex);
      predictions = predictions.slice(-stablePredictionCount);

      if (predictions.length === stablePredictionCount) {
        const isStable = predictions.every(
          (prediction) => prediction === actionIndex,
        );
        if (isStable && confidence > threshold) {
          const predictedAction = actions[actionIndex];
          if (
            sentence.length === 0 ||
            sentence[sentence.length - 1] !== predictedAction
          ) {
            sentence.push(predictedAction);
            sentence = sentence.slice(-5);
          }
        }
      }

      const recognizedText =
        sentence.length > 0 ? sentence.join(" ") : "Waiting for signs...";
      sentenceElement.textContent = recognizedText;
      statusElement.textContent = recognizedText;
    } catch (error) {
      console.error("Prediction error:", error);
      statusElement.textContent = "Prediction service unavailable. Try again.";
    } finally {
      isPredicting = false;
    }
  }
});
