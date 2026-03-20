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
  const stabilityCount = 10;
  const modelUrl = (window.APP_CONFIG?.modelUrl || "").trim();
  const apiBaseUrl = (window.APP_CONFIG?.apiBaseUrl || "").replace(/\/$/, "");
  const useBackendFallback = window.APP_CONFIG?.useBackendFallback !== false;

  let sequence = [];
  let predictions = [];
  let sentence = [];
  let isRunning = false;
  let isPredicting = false;
  let model = null;
  let holistic = null;
  let camera = null;
  let latestConfidence = null;

  stopBtn.disabled = true;
  resetBtn.disabled = true;
  statusElement.textContent = "Waiting for signs...";

  startBtn.addEventListener("click", startRecognition);
  stopBtn.addEventListener("click", stopRecognition);
  resetBtn.addEventListener("click", resetRecognition);

  async function startRecognition() {
    if (isRunning) return;

    startBtn.disabled = true;
    statusElement.textContent = "Preparing model and camera...";

    await setupModelAndDetector();

    if (!holistic || (!model && !useBackendFallback)) {
      startBtn.disabled = false;
      return;
    }

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
    } catch (error) {
      console.error("Failed to start camera:", error);
      statusElement.textContent = "Camera permission denied or unavailable.";
      cameraStateElement.textContent = "Camera unavailable";
      startBtn.disabled = false;
      return;
    }

    isRunning = true;
    stopBtn.disabled = false;
    resetBtn.disabled = false;
    cameraStateElement.textContent = "Camera is live";
    statusElement.textContent = "Collecting landmarks...";
  }

  function stopRecognition() {
    if (!isRunning) return;

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
    latestConfidence = null;
    isPredicting = false;

    sentenceElement.textContent = "";
    confidenceElement.textContent = "Confidence: --";
    statusElement.textContent = isRunning
      ? "Collecting landmarks..."
      : "Waiting for signs...";
  }

  async function setupModelAndDetector() {
    if (!holistic) {
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

    if (!model && modelUrl) {
      try {
        model = await tf.loadLayersModel(modelUrl);
      } catch (error) {
        console.warn("Browser model load failed, fallback may be used:", error);
      }
    }
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

    drawOverlay();
    canvasCtx.restore();

    const keypoints = extractKeypoints(results);
    sequence.push(keypoints);
    sequence = sequence.slice(-sequenceLength);

    if (sequence.length < sequenceLength || isPredicting) {
      if (sequence.length < sequenceLength) {
        statusElement.textContent = `Collecting landmarks... ${sequence.length}/${sequenceLength}`;
      }
      return;
    }

    runPrediction(sequence);
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
    for (let index = 0; index < landmarks.length; index += 1) {
      const landmark = landmarks[index];
      flattened.push(landmark.x, landmark.y, landmark.z);
      if (includeVisibility) {
        flattened.push(
          typeof landmark.visibility === "number" ? landmark.visibility : 0,
        );
      }
    }
    return flattened;
  }

  async function runPrediction(currentSequence) {
    isPredicting = true;
    try {
      let predictedIndex = -1;
      let confidence = 0;

      if (model) {
        const inputTensor = tf.tensor(
          currentSequence,
          [1, sequenceLength, 1662],
          "float32",
        );
        const outputTensor = model.predict(inputTensor);
        const probabilities = await outputTensor.data();

        predictedIndex = probabilities.indexOf(Math.max(...probabilities));
        confidence = Number(probabilities[predictedIndex] || 0);

        inputTensor.dispose();
        outputTensor.dispose();
      } else if (useBackendFallback) {
        const response = await fetch(`${apiBaseUrl}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sequence: currentSequence }),
        });

        if (!response.ok) {
          const message = await response.text();
          throw new Error(
            `Fallback prediction failed ${response.status}: ${message}`,
          );
        }

        const data = await response.json();
        predictedIndex = actions.indexOf(data.action);
        confidence = Number(data.confidence || 0);
      }

      if (predictedIndex < 0) {
        return;
      }

      latestConfidence = confidence;
      confidenceElement.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;

      predictions.push(predictedIndex);
      predictions = predictions.slice(-stabilityCount);

      if (predictions.length === stabilityCount) {
        const stable = predictions.every((value) => value === predictedIndex);
        if (stable && confidence > threshold) {
          const detectedAction = actions[predictedIndex];
          if (
            sentence.length === 0 ||
            sentence[sentence.length - 1] !== detectedAction
          ) {
            sentence.push(detectedAction);
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
      latestConfidence = null;
      statusElement.textContent =
        "Prediction unavailable. Check model/fallback setup.";
    } finally {
      isPredicting = false;
    }
  }

  function drawOverlay() {
    const recognizedText =
      sentence.length > 0 ? sentence.join(" ") : "Waiting for signs...";
    const confidenceText =
      latestConfidence === null
        ? "Confidence: --"
        : `Confidence: ${(latestConfidence * 100).toFixed(1)}%`;

    canvasCtx.fillStyle = "rgba(245, 117, 16, 0.9)";
    canvasCtx.fillRect(0, 0, canvasElement.width, 46);
    canvasCtx.fillStyle = "#ffffff";
    canvasCtx.font = "20px Arial";
    canvasCtx.fillText(recognizedText, 10, 30);

    canvasCtx.fillStyle = "rgba(0, 0, 0, 0.5)";
    canvasCtx.fillRect(0, canvasElement.height - 34, 220, 34);
    canvasCtx.fillStyle = "#ffffff";
    canvasCtx.font = "15px Arial";
    canvasCtx.fillText(confidenceText, 10, canvasElement.height - 12);
  }
});
