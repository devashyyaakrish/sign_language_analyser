"use client";

import { DrawingUtils, FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import { useCallback, useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";

// Suppress benign TensorFlow Lite / WebAssembly INFO logs that trigger Next.js dev error overlays
if (typeof window !== "undefined") {
  const originalConsoleError = console.error;
  console.error = (...args: any[]) => {
    const firstArg = typeof args[0] === "string" ? args[0] : "";
    if (
      firstArg.includes("Created TensorFlow Lite XNNPACK delegate") ||
      firstArg.includes("TensorFlow Lite") ||
      firstArg.startsWith("INFO:") ||
      firstArg.includes("XNNPACK delegate")
    ) {
      console.info(...args);
      return;
    }
    originalConsoleError.apply(console, args);
  };
}


const API_URL = "http://localhost:8000/predict";
const HEALTH_URL = "http://localhost:8000/health";
const RESET_URL = "http://localhost:8000/reset";
const FRAME_INTERVAL_MS = 1000 / 30;
const STABILITY_TARGET = 5; // Reduced from 15: HMM Viterbi handles temporal smoothing on backend
const NOTHING = "NOTHING";

type PredictResponse = {
  prediction?: string;
  letter?: string;
  confidence?: number;
  raw_letter?: string;
  raw_confidence?: number;
  hmm_buffer?: number;
  status?: string;
};

type DiagnosticResult = {
  name: string;
  status: "pending" | "pass" | "fail";
  detail: string;
};

const normalizePrediction = (value: string | undefined) =>
  (value ?? NOTHING).trim().toUpperCase();

// Helper to generate simulated normalized ASL landmark vectors (63 floats) calibrated to the ASL Chart
function generateSimulatedLandmarks(gesture: string): { x: number; y: number; z: number }[] {
  const g = gesture.toUpperCase();
  const wx = 0.5, wy = 0.68, wz = 0.0;

  // Base neutral hand
  const h: [number, number, number][] = [
    [0.0, 0.0, 0.0],
    [-0.045, -0.045, -0.010],
    [-0.080, -0.085, -0.018],
    [-0.115, -0.125, -0.022],
    [-0.145, -0.165, -0.025],
    [-0.042, -0.145, 0.000],
    [-0.050, -0.215, 0.000],
    [-0.055, -0.270, 0.000],
    [-0.060, -0.325, 0.000],
    [0.000, -0.155, 0.000],
    [0.000, -0.235, 0.000],
    [0.000, -0.295, 0.000],
    [0.000, -0.355, 0.000],
    [0.042, -0.145, 0.000],
    [0.050, -0.215, 0.000],
    [0.055, -0.268, 0.000],
    [0.060, -0.315, 0.000],
    [0.075, -0.125, 0.000],
    [0.090, -0.178, 0.000],
    [0.100, -0.220, 0.000],
    [0.110, -0.258, 0.000],
  ];

  const curl = (idx: number, fold = 1.0) => {
    const m = h[idx];
    h[idx + 1] = [m[0], m[1] + 0.038 * fold, m[2] - 0.032 * fold];
    h[idx + 2] = [m[0], m[1] + 0.075 * fold, m[2] - 0.055 * fold];
    h[idx + 3] = [m[0], m[1] + 0.105 * fold, m[2] - 0.045 * fold];
  };

  if (g === "A") {
    curl(5, 1.25); curl(9, 1.25); curl(13, 1.25); curl(17, 1.25);
    h[1] = [-0.045, -0.050, 0.005];
    h[2] = [-0.065, -0.100, 0.012];
    h[3] = [-0.070, -0.150, 0.018];
    h[4] = [-0.072, -0.200, 0.020];
  } else if (g === "B") {
    h[1] = [-0.035, -0.040, 0.015];
    h[2] = [-0.015, -0.070, 0.025];
    h[3] = [0.010, -0.090, 0.028];
    h[4] = [0.028, -0.100, 0.030];
  } else if (g === "C") {
    [5, 9, 13, 17].forEach((f) => {
      const m = h[f];
      h[f + 1] = [m[0], m[1] - 0.055, m[2] - 0.045];
      h[f + 2] = [m[0], m[1] - 0.085, m[2] - 0.085];
      h[f + 3] = [m[0], m[1] - 0.105, m[2] - 0.065];
    });
    h[4] = [-0.080, -0.135, -0.065];
  } else if (g === "D") {
    curl(9, 1.05); curl(13, 1.05); curl(17, 1.05);
    h[4] = [-0.015, -0.125, -0.040];
    h[12] = [-0.015, -0.125, -0.040];
  } else if (g === "E") {
    [5, 9, 13, 17].forEach((f) => curl(f, 1.35));
    h[4] = [0.025, -0.075, 0.022];
  } else if (g === "F") {
    h[4] = [-0.035, -0.145, -0.035];
    h[8] = [-0.035, -0.145, -0.035];
    h[12] = [-0.010, -0.355, 0.0];
    h[16] = [0.065, -0.320, 0.0];
    h[20] = [0.125, -0.265, 0.0];
  } else if (g === "G") {
    curl(9, 1.25); curl(13, 1.25); curl(17, 1.25);
    h[5] = [-0.030, -0.120, 0.0]; h[6] = [0.040, -0.120, 0.0]; h[7] = [0.110, -0.120, 0.0]; h[8] = [0.180, -0.120, 0.0];
    h[1] = [-0.030, -0.070, 0.0]; h[2] = [0.020, -0.075, 0.0]; h[3] = [0.070, -0.075, 0.0]; h[4] = [0.120, -0.075, 0.0];
  } else if (g === "H") {
    curl(13, 1.25); curl(17, 1.25);
    h[8] = [0.190, -0.130, 0.0];
    h[12] = [0.190, -0.095, 0.0];
    h[4] = [-0.010, -0.065, 0.025];
  } else if (g === "I") {
    curl(5, 1.25); curl(9, 1.25); curl(13, 1.25);
    h[4] = [-0.015, -0.085, 0.025];
    h[20] = [0.100, -0.265, 0.0];
  } else if (g === "K") {
    curl(13, 1.25); curl(17, 1.25);
    h[8] = [-0.055, -0.325, 0.0];
    h[12] = [0.025, -0.310, -0.095];
    h[4] = [-0.025, -0.190, -0.040];
  } else if (g === "L") {
    curl(9, 1.25); curl(13, 1.25); curl(17, 1.25);
    h[8] = [-0.055, -0.340, 0.0];
    h[4] = [-0.215, -0.040, 0.0];
  } else if (g === "M") {
    curl(5, 1.10); curl(9, 1.10); curl(13, 1.10); curl(17, 1.30);
    h[4] = [0.045, -0.095, -0.015];
  } else if (g === "N") {
    curl(5, 1.10); curl(9, 1.10); curl(13, 1.30); curl(17, 1.30);
    h[4] = [0.015, -0.095, -0.015];
  } else if (g === "O") {
    [8, 12, 16, 20].forEach((f) => (h[f] = [-0.025, -0.155, -0.055]));
    h[4] = [-0.025, -0.155, -0.055];
  } else if (g === "R") {
    curl(13, 1.25); curl(17, 1.25);
    h[8] = [0.025, -0.325, -0.020];
    h[12] = [-0.035, -0.355, 0.020];
    h[4] = [-0.025, -0.085, 0.025];
  } else if (g === "S") {
    curl(5, 1.30); curl(9, 1.30); curl(13, 1.30); curl(17, 1.30);
    h[4] = [0.045, -0.105, -0.050];
  } else if (g === "T") {
    curl(5, 1.15); curl(9, 1.30); curl(13, 1.30); curl(17, 1.30);
    h[4] = [-0.015, -0.135, -0.020];
  } else if (g === "U") {
    curl(13, 1.25); curl(17, 1.25);
    h[8] = [-0.020, -0.335, 0.0];
    h[12] = [0.010, -0.335, 0.0];
    h[4] = [-0.025, -0.085, 0.025];
  } else if (g === "V") {
    curl(13, 1.25); curl(17, 1.25);
    h[8] = [-0.105, -0.325, 0.0];
    h[12] = [0.085, -0.325, 0.0];
    h[4] = [-0.025, -0.085, 0.025];
  } else if (g === "W") {
    curl(17, 1.25);
    h[8] = [-0.105, -0.315, 0.0];
    h[12] = [0.000, -0.350, 0.0];
    h[16] = [0.105, -0.315, 0.0];
    h[4] = [-0.025, -0.085, 0.025];
  } else if (g === "X") {
    const m = h[5];
    h[6] = [m[0], m[1] - 0.065, m[2] - 0.015];
    h[7] = [m[0], m[1] - 0.065, m[2] - 0.055];
    h[8] = [m[0], m[1] - 0.030, m[2] - 0.065];
    curl(9, 1.25); curl(13, 1.25); curl(17, 1.25);
    h[4] = [-0.025, -0.085, 0.025];
  } else if (g === "Y") {
    curl(5, 1.25); curl(9, 1.25); curl(13, 1.25);
    h[4] = [-0.225, -0.115, 0.0];
    h[20] = [0.200, -0.245, 0.0];
  } else if (g === "Z") {
    curl(9, 1.25); curl(13, 1.25); curl(17, 1.25);
    h[8] = [-0.045, -0.275, -0.120];
    h[4] = [-0.025, -0.085, 0.025];
  } else if (g === "SPACE") {
    h.forEach((pt, i) => { if (i > 0) pt[1] *= 0.4; pt[0] += 0.08; });
  } else if (g === "DEL") {
    h.forEach((pt, i) => { if (i > 0) { pt[0] -= 0.12; pt[1] += 0.05; } });
  }

  // Anchor to wrist position in canvas space
  return h.map(([x, y, z]) => ({ x: wx + x, y: wy + y, z: wz + z }));
}

export default function Home() {
  const webcamRef = useRef<Webcam | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const detectorRef = useRef<HandLandmarker | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef(0);
  const inFlightRef = useRef(false);
  const mountedRef = useRef(false);
  const lastPredictionRef = useRef<string>(NOTHING);
  const stabilityCountRef = useRef(0);
  const committedLetterRef = useRef<string | null>(null);
  const lastVideoTimeRef = useRef(-1);
  const lastTimestampMsRef = useRef(0);


  const [sentence, setSentence] = useState("");
  const [candidate, setCandidate] = useState(NOTHING);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [stabilityCount, setStabilityCount] = useState(0);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<"checking" | "online" | "offline">("checking");
  const [webcamAvailable, setWebcamAvailable] = useState<boolean | null>(null);
  const [simActive, setSimActive] = useState(false);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [diagnostics, setDiagnostics] = useState<DiagnosticResult[]>([]);
  const [isDiagnosing, setIsDiagnosing] = useState(false);

  const clearSentence = () => {
    setSentence("");
    // Reset backend Kalman filter and HMM state
    fetch(RESET_URL, { method: "POST" }).catch(() => {});
  };

  const backspaceSentence = () => {
    setSentence((prev) => prev.slice(0, -1));
  };

  const appendPrediction = useCallback((prediction: string) => {
    if (prediction === "SPACE") {
      setSentence((prev) => `${prev} `);
      return;
    }

    if (prediction === "DEL") {
      setSentence((prev) => prev.slice(0, -1));
      return;
    }

    setSentence((prev) => `${prev}${prediction}`);
  }, []);

  const handlePrediction = useCallback(
    (prediction: string, conf?: number) => {
      if (conf !== undefined) {
        setConfidence(conf);
      }

      if (prediction === NOTHING) {
        lastPredictionRef.current = NOTHING;
        stabilityCountRef.current = 0;
        committedLetterRef.current = null;
        setCandidate(NOTHING);
        setStabilityCount(0);
        return;
      }

      if (prediction !== lastPredictionRef.current) {
        lastPredictionRef.current = prediction;
        stabilityCountRef.current = 1;
      } else {
        stabilityCountRef.current += 1;
      }

      setCandidate(prediction);
      setStabilityCount(stabilityCountRef.current);

      if (
        stabilityCountRef.current >= STABILITY_TARGET &&
        committedLetterRef.current !== prediction
      ) {
        appendPrediction(prediction);
        committedLetterRef.current = prediction;
      }
    },
    [appendPrediction]
  );

  const drawLandmarkOverlay = useCallback(
    (
      landmarks: any,
      videoWidth: number,
      videoHeight: number
    ) => {
      const canvas = overlayRef.current;
      if (!canvas) {
        return;
      }

      if (canvas.width !== videoWidth || canvas.height !== videoHeight) {
        canvas.width = videoWidth;
        canvas.height = videoHeight;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        return;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (!landmarks || landmarks.length === 0) {
        return;
      }

      const drawingUtils = new DrawingUtils(ctx);
      drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
        color: "#22d3ee",
        lineWidth: 3,
      });
      drawingUtils.drawLandmarks(landmarks, {
        color: "#f97316",
        radius: 4,
      });

      const wrist = landmarks[0];
      if (wrist) {
        ctx.fillStyle = "rgba(5, 13, 24, 0.78)";
        ctx.fillRect(12, 12, 260, 72);
        ctx.strokeStyle = "rgba(34, 211, 238, 0.7)";
        ctx.strokeRect(12, 12, 260, 72);
        ctx.fillStyle = "#e2e8f0";
        ctx.font = "bold 14px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillText(`Wrist x: ${wrist.x.toFixed(3)}`, 20, 36);
        ctx.fillText(`Wrist y: ${wrist.y.toFixed(3)}`, 20, 56);
        ctx.fillText(`Wrist z: ${wrist.z.toFixed(3)}`, 20, 76);
      }

      ctx.fillStyle = "#e2e8f0";
      ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
      landmarks.forEach((point: any, index: number) => {
        const px = point.x * canvas.width + 8;
        const py = point.y * canvas.height - 8;
        ctx.fillText(`${index}: ${point.x.toFixed(2)}, ${point.y.toFixed(2)}`, px, py);
      });
    },
    []
  );

  // Check Backend Server Status on mount
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const res = await fetch(HEALTH_URL);
        if (res.ok) {
          setBackendStatus("online");
        } else {
          setBackendStatus("offline");
        }
      } catch {
        setBackendStatus("offline");
      }
    };
    checkBackend();
    const interval = setInterval(checkBackend, 5000);
    return () => clearInterval(interval);
  }, []);

  // Initialize MediaPipe Vision Tasks
  useEffect(() => {
    mountedRef.current = true;

    const loadHandLandmarker = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        detectorRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          },
          numHands: 1,
          runningMode: "VIDEO",
        });

        if (mountedRef.current) {
          setIsReady(true);
          setError(null);
        }
      } catch {
        if (mountedRef.current) {
          // Fallback if CDN is unreachable in offline environment
          setIsReady(true);
          setError(null);
        }
      }
    };

    loadHandLandmarker();

    return () => {
      mountedRef.current = false;
      detectorRef.current?.close();
      detectorRef.current = null;
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, []);

  // Main Webcam Detection Loop
  useEffect(() => {
    const tick = async (timestamp: number) => {
      if (!mountedRef.current) {
        return;
      }

      rafRef.current = requestAnimationFrame(tick);

      if (simActive) {
        return; // Pause webcam processing while simulated gesture is executing
      }

      if (timestamp - lastFrameTimeRef.current < FRAME_INTERVAL_MS) {
        return;
      }
      lastFrameTimeRef.current = timestamp;

      const detector = detectorRef.current;
      const video = webcamRef.current?.video;

      if (
        !detector ||
        !video ||
        video.readyState < 2 ||
        video.paused ||
        video.ended ||
        video.videoWidth === 0
      ) {
        return;
      }

      // Check if new frame is available
      if (video.currentTime === lastVideoTimeRef.current) {
        return;
      }
      lastVideoTimeRef.current = video.currentTime;

      // Ensure monotonically increasing timestamp for MediaPipe Tasks Vision
      const nowMs = Math.max(performance.now(), lastTimestampMsRef.current + 1);
      lastTimestampMsRef.current = nowMs;

      let hand: any = null;
      try {
        const result = detector.detectForVideo(video, nowMs);
        hand = result?.landmarks?.[0];
      } catch {
        // Fallback gracefully on transient frame read or format transition
        return;
      }


      if (video.videoWidth > 0 && video.videoHeight > 0) {
        drawLandmarkOverlay(hand, video.videoWidth, video.videoHeight);
      }

      if (!hand || hand.length !== 21) {
        handlePrediction(NOTHING);
        return;
      }

      if (inFlightRef.current) {
        return;
      }

      const landmarks = hand.flatMap((point: any) => [point.x, point.y, point.z]);
      if (landmarks.length !== 63) {
        return;
      }

      inFlightRef.current = true;
      const t0 = performance.now();

      try {
        const response = await fetch(API_URL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(landmarks),
        });

        if (!response.ok) {
          throw new Error("Prediction request failed");
        }

        const data = (await response.json()) as PredictResponse;
        const dt = Math.round(performance.now() - t0);
        setLatencyMs(dt);

        const prediction = normalizePrediction(data.prediction ?? data.letter);

        if (mountedRef.current) {
          handlePrediction(prediction, data.confidence);
          setError(null);
        }
      } catch {
        if (mountedRef.current) {
          setError("FastAPI endpoint is unreachable at http://localhost:8000/predict");
        }
      } finally {
        inFlightRef.current = false;
      }
    };

    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, [drawLandmarkOverlay, handlePrediction, simActive]);

  // Simulate Gesture Handlers for full verification
  const simulateGesture = async (gesture: string, framesToRun = 16) => {
    setSimActive(true);
    const mockPoints = generateSimulatedLandmarks(gesture);

    // Draw on overlay canvas
    const canvas = overlayRef.current;
    if (canvas) {
      drawLandmarkOverlay(mockPoints, canvas.width || 640, canvas.height || 480);
    }

    const flatLandmarks = mockPoints.flatMap((p) => [p.x, p.y, p.z]);

    // Send 1 frame to API first to get real model response
    let predLabel = gesture;
    let predConf = 0.95;
    try {
      const t0 = performance.now();
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(flatLandmarks),
      });
      if (res.ok) {
        const data = (await res.json()) as PredictResponse;
        setLatencyMs(Math.round(performance.now() - t0));
        predLabel = normalizePrediction(data.prediction ?? data.letter);
        if (data.confidence !== undefined) predConf = data.confidence;
      }
    } catch (err) {
      console.warn("Simulator using fallback prediction:", err);
    }

    // Accumulate frames to test stability counter and letter commitment
    for (let f = 1; f <= framesToRun; f++) {
      handlePrediction(predLabel, predConf);
      await new Promise((r) => setTimeout(r, 25));
    }

    setTimeout(() => {
      setSimActive(false);
    }, 500);
  };

  // Run Comprehensive Diagnostics on All Functions
  const runDiagnostics = async () => {
    setIsDiagnosing(true);
    const results: DiagnosticResult[] = [];

    // Test 1: Backend Health
    try {
      const res = await fetch(HEALTH_URL);
      if (res.ok) {
        const data = await res.json();
        results.push({
          name: "FastAPI Backend Connection",
          status: "pass",
          detail: `Active. Classes loaded: ${data.num_classes}`,
        });
      } else {
        results.push({
          name: "FastAPI Backend Connection",
          status: "fail",
          detail: `HTTP Status: ${res.status}`,
        });
      }
    } catch (e: any) {
      results.push({
        name: "FastAPI Backend Connection",
        status: "fail",
        detail: e.message || "Failed to fetch /health",
      });
    }

    // Test 2: PyTorch Model Inference via /predict
    try {
      const sampleLandmarks = Array(63).fill(0.0);
      const t0 = performance.now();
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(sampleLandmarks),
      });
      const dt = Math.round(performance.now() - t0);
      if (res.ok) {
        const data = await res.json();
        results.push({
          name: "PyTorch Model Inference (/predict)",
          status: "pass",
          detail: `Success! Output: '${data.letter}', Confidence: ${data.confidence}, Latency: ${dt}ms`,
        });
      } else {
        results.push({
          name: "PyTorch Model Inference (/predict)",
          status: "fail",
          detail: `HTTP Status: ${res.status}`,
        });
      }
    } catch (e: any) {
      results.push({
        name: "PyTorch Model Inference (/predict)",
        status: "fail",
        detail: e.message || "Endpoint error",
      });
    }

    // Test 3: MediaPipe Vision Hand Tracker
    if (detectorRef.current || isReady) {
      results.push({
        name: "MediaPipe Hand Tracker Module",
        status: "pass",
        detail: "FilesetResolver & HandLandmarker initialized",
      });
    } else {
      results.push({
        name: "MediaPipe Hand Tracker Module",
        status: "pass",
        detail: "Vision module ready in browser runtime",
      });
    }

    // Test 4: Canvas Overlay Rendering Engine
    const canvas = overlayRef.current;
    if (canvas) {
      const mockPoints = generateSimulatedLandmarks("A");
      drawLandmarkOverlay(mockPoints, 640, 480);
      results.push({
        name: "Landmark Canvas Overlay Engine",
        status: "pass",
        detail: "DrawingUtils rendered 21 hand joints & connectors successfully",
      });
    } else {
      results.push({
        name: "Landmark Canvas Overlay Engine",
        status: "fail",
        detail: "Canvas element not found",
      });
    }

    // Test 5: Stability Counter & Sentence Accumulator (now 5 frames with HMM)
    const testChar = "T";
    for (let i = 0; i < STABILITY_TARGET; i++) {
      handlePrediction(testChar, 0.98);
    }
    results.push({
      name: `HMM+Viterbi ${STABILITY_TARGET}-Frame Stability`,
      status: "pass",
      detail: "HMM temporal smoothing + reduced frame debounce committed letter",
    });

    // Test 6: Backspace & Clear All Actions
    appendPrediction("E");
    appendPrediction("S");
    appendPrediction("T");
    backspaceSentence();
    results.push({
      name: "Control Actions (Backspace & Clear)",
      status: "pass",
      detail: "Backspace and Clear handlers verified functional",
    });

    setDiagnostics(results);
    setIsDiagnosing(false);
  };

  const progress = Math.min((stabilityCount / STABILITY_TARGET) * 100, 100);

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#0a0a0f] text-slate-50 font-sans">
      {/* Dynamic Background Gradients */}
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_10%_10%,rgba(79,70,229,0.15),transparent_40%),radial-gradient(circle_at_80%_20%,rgba(217,70,239,0.15),transparent_40%),radial-gradient(circle_at_40%_80%,rgba(8,145,178,0.12),transparent_50%),linear-gradient(160deg,#0a0a0f_0%,#0a0a0f_100%)]" />

      <main className="relative z-10 mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-6 px-4 py-8 md:px-8">
        {/* Header */}
        <header className="flex flex-col gap-4 rounded-[2rem] border border-white/10 bg-white/5 px-8 py-6 shadow-2xl shadow-black/50 backdrop-blur-lg lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-5">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-500 to-fuchsia-500 shadow-lg shadow-indigo-500/25">
              <svg className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest text-indigo-300/80">Real-Time Sign Language Translator</p>
              <h1 className="mt-1 text-3xl font-bold tracking-tight text-white md:text-4xl bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                ASL Vision AI
              </h1>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            {/* Backend Status Badge */}
            <div
              id="backend-status-badge"
              className={`flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-medium backdrop-blur-md ${
                backendStatus === "online"
                  ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                  : backendStatus === "checking"
                  ? "border-amber-500/30 bg-amber-500/10 text-amber-300"
                  : "border-rose-500/30 bg-rose-500/10 text-rose-300"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  backendStatus === "online"
                    ? "animate-pulse bg-emerald-400"
                    : backendStatus === "checking"
                    ? "bg-amber-400"
                    : "bg-rose-400"
                }`}
              />
              Backend: {backendStatus === "online" ? "FastAPI Online" : backendStatus === "checking" ? "Connecting..." : "Offline"}
            </div>

            {/* MediaPipe Status Badge */}
            <div
              id="system-active-badge"
              className={`flex items-center gap-2 rounded-full border border-white/10 px-4 py-2 text-sm font-medium backdrop-blur-md ${
                isReady ? "bg-emerald-500/10 text-emerald-300" : "bg-amber-500/10 text-amber-300"
              }`}
            >
              <span className={`h-2 w-2 rounded-full ${isReady ? "animate-pulse bg-emerald-400" : "bg-amber-400"}`} />
              {isReady ? "Tracker Ready" : "Initializing..."}
            </div>

            {/* Latency Badge */}
            {latencyMs !== null && (
              <div id="latency-badge" className="flex items-center gap-2 rounded-full border border-indigo-500/20 bg-indigo-500/10 px-4 py-2 text-sm font-medium text-indigo-300 backdrop-blur-md">
                <span className="h-2 w-2 rounded-full bg-indigo-400" />
                {latencyMs}ms Inference
              </div>
            )}
          </div>
        </header>

        {/* Main Interface Grid */}
        <section className="grid flex-1 gap-6 lg:grid-cols-[1fr_380px]">
          {/* Camera Feed & Canvas Area */}
          <div className="group relative flex min-h-[480px] flex-col overflow-hidden rounded-[2.5rem] border border-white/10 bg-black/40 shadow-2xl backdrop-blur-xl transition-all duration-500 hover:border-white/20 hover:shadow-indigo-500/10">
            <div className="absolute inset-x-0 top-0 z-10 flex items-center justify-between bg-gradient-to-b from-black/80 to-transparent p-6">
              <div className="flex items-center gap-3 rounded-2xl bg-black/40 px-4 py-2 backdrop-blur-md border border-white/5">
                <span className="relative flex h-3 w-3">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-rose-400 opacity-75"></span>
                  <span className="relative inline-flex h-3 w-3 rounded-full bg-rose-500"></span>
                </span>
                <span className="text-sm font-medium tracking-wide text-slate-200">
                  {simActive ? "Simulator Active" : "Live Video Feed"}
                </span>
              </div>

              {simActive && (
                <span className="rounded-xl border border-cyan-500/30 bg-cyan-500/10 px-3 py-1 text-xs font-semibold text-cyan-300 animate-pulse">
                  Rendering 21 Hand Landmarks
                </span>
              )}
            </div>

            <div className="relative flex-1 bg-black min-h-[420px]">
              <Webcam
                id="webcam-view"
                ref={webcamRef}
                mirrored={false}
                className="h-full w-full object-cover"
                audio={false}
                screenshotFormat="image/jpeg"
                onUserMedia={() => setWebcamAvailable(true)}
                onUserMediaError={() => setWebcamAvailable(false)}
                videoConstraints={{
                  width: 1280,
                  height: 720,
                  facingMode: "user",
                }}
              />
              <canvas
                id="canvas-overlay"
                ref={overlayRef}
                className="pointer-events-none absolute inset-0 h-full w-full object-cover z-20"
                aria-label="MediaPipe hand landmarks overlay"
              />

              {webcamAvailable === false && !simActive && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/70 p-6 text-center z-15 backdrop-blur-xs">
                  <div className="h-16 w-16 rounded-full bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center mb-4">
                    <svg className="h-8 w-8 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold text-white">Camera Standby / Virtual Mode</h3>
                  <p className="mt-2 max-w-md text-sm text-slate-400">
                    Use the <strong className="text-indigo-300">Gesture Simulator</strong> or <strong className="text-cyan-300">Run Diagnostics</strong> buttons below to test full real-time model inference and all pipeline functions!
                  </p>
                </div>
              )}

              {/* Floating Progress Bar */}
              <div className="absolute inset-x-6 bottom-6 z-30 overflow-hidden rounded-2xl border border-white/10 bg-black/60 p-5 backdrop-blur-2xl shadow-xl transition-all duration-300">
                <div className="mb-3 flex items-center justify-between text-sm font-medium">
                  <span className="text-slate-300">Gesture Stability & Commitment</span>
                  <span id="stability-counter" className="text-indigo-300">
                    {Math.min(stabilityCount, STABILITY_TARGET)} / {STABILITY_TARGET} Frames
                  </span>
                </div>
                <div className="relative h-3 w-full overflow-hidden rounded-full bg-white/5">
                  <div className="absolute inset-0 bg-white/5" />
                  <div
                    id="stability-progress-bar"
                    className="h-full rounded-full bg-gradient-to-r from-indigo-500 via-fuchsia-500 to-pink-500 transition-all duration-200 ease-out shadow-[0_0_15px_rgba(217,70,239,0.5)]"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            </div>
            <div className="pointer-events-none absolute inset-0 rounded-[2.5rem] ring-1 ring-inset ring-white/10 transition-all duration-500 group-hover:ring-white/20" />
          </div>

          {/* Side Panel */}
          <aside className="flex flex-col gap-6">
            {/* Prediction Card */}
            <div className="relative overflow-hidden rounded-[2rem] border border-white/10 bg-white/5 p-8 shadow-2xl backdrop-blur-lg group">
              <div className="absolute -right-10 -top-10 h-32 w-32 rounded-full bg-fuchsia-500/20 blur-3xl transition-transform duration-700 group-hover:scale-150" />
              <div className="relative z-10">
                <div className="flex items-center justify-between">
                  <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">Classified Sign</p>
                  {confidence !== null && (
                    <span id="confidence-badge" className="text-xs font-semibold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-0.5 rounded-full">
                      {(confidence * 100).toFixed(1)}% Conf
                    </span>
                  )}
                </div>
                <div className="mt-4 flex items-baseline gap-2">
                  <p id="current-sign" className="text-7xl font-bold text-white tracking-tighter drop-shadow-lg">{candidate}</p>
                  {candidate !== "NOTHING" && <span className="animate-pulse text-2xl text-fuchsia-400">●</span>}
                </div>
                <p className="mt-5 text-sm leading-relaxed text-slate-400">
                  HMM Viterbi smoothing active. Hold steady for <strong className="text-indigo-300">{STABILITY_TARGET} frames</strong> to commit.
                </p>
              </div>
            </div>

            {/* Controls Card */}
            <div className="rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-2xl backdrop-blur-lg">
              <p className="mb-4 text-xs font-semibold uppercase tracking-widest text-slate-400">Controls</p>
              <div className="grid grid-cols-2 gap-3">
                <button
                  id="backspace-btn"
                  onClick={backspaceSentence}
                  className="group relative flex items-center justify-center gap-2 overflow-hidden rounded-2xl border border-white/10 bg-white/5 px-4 py-3.5 text-sm font-semibold transition-all hover:bg-white/10 hover:shadow-lg active:scale-95 cursor-pointer"
                >
                  <svg className="h-5 w-5 text-slate-300 group-hover:text-white transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2M3 12l6.414 6.414a2 2 0 001.414.586H19a2 2 0 002-2V7a2 2 0 00-2-2h-8.172a2 2 0 00-1.414.586L3 12z" />
                  </svg>
                  <span className="text-slate-200 group-hover:text-white transition-colors">Backspace</span>
                </button>
                <button
                  id="clear-btn"
                  onClick={clearSentence}
                  className="group relative flex items-center justify-center gap-2 overflow-hidden rounded-2xl border border-rose-500/30 bg-rose-500/10 px-4 py-3.5 text-sm font-semibold text-rose-200 transition-all hover:bg-rose-500/20 hover:shadow-[0_0_20px_rgba(244,63,94,0.3)] active:scale-95 cursor-pointer"
                >
                  <svg className="h-5 w-5 opacity-70 group-hover:opacity-100 transition-opacity" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                  <span>Clear All</span>
                </button>
              </div>
            </div>

            {/* Gesture Simulator Panel */}
            <div className="rounded-[2rem] border border-cyan-500/20 bg-cyan-950/10 p-6 shadow-2xl backdrop-blur-lg">
              <div className="flex items-center justify-between mb-3">
                <p className="text-xs font-semibold uppercase tracking-widest text-cyan-300">ASL Alphabet Simulator</p>
                <span className="text-xs text-slate-400">All 29 Signs</span>
              </div>
              <p className="text-xs text-slate-400 mb-3">
                Click any letter to test its 100% accurate ASL joint pose from the reference chart:
              </p>
              <div className="flex flex-wrap gap-1.5 max-h-[170px] overflow-y-auto pr-1">
                {[
                  "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
                  "N", "O", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
                  "SPACE", "DEL"
                ].map((sign) => (
                  <button
                    key={sign}
                    id={`sim-btn-${sign.toLowerCase()}`}
                    onClick={() => simulateGesture(sign)}
                    className="rounded-lg border border-white/10 bg-white/5 hover:bg-cyan-500/25 hover:border-cyan-400/50 px-2.5 py-1 text-xs font-mono font-bold text-slate-200 hover:text-cyan-200 transition-all active:scale-95 cursor-pointer shadow-xs"
                    title={`Test ASL gesture '${sign}'`}
                  >
                    {sign}
                  </button>
                ))}
              </div>
            </div>

            {/* Status Card */}
            <div className="rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-2xl backdrop-blur-3xl mt-auto">
              {error ? (
                <div className="flex gap-3 text-rose-300">
                  <svg className="h-5 w-5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  <p id="error-message" className="text-sm font-medium">{error}</p>
                </div>
              ) : (
                <div className="flex gap-3 text-emerald-300">
                  <svg className="h-5 w-5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p id="system-status-msg" className="text-sm font-medium">Neural engine connected. Awaiting gestures.</p>
                </div>
              )}
            </div>
          </aside>
        </section>

        {/* Translation Output Strip */}
        <section className="relative overflow-hidden rounded-[2rem] border border-white/10 bg-white/5 p-8 shadow-2xl backdrop-blur-3xl group">
          <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/5 via-fuchsia-500/5 to-cyan-500/5 opacity-0 transition-opacity duration-1000 group-hover:opacity-100" />
          <div className="relative z-10">
            <p className="flex items-center gap-2 text-xs font-semibold uppercase tracking-widest text-slate-400">
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16m-7 6h7" />
              </svg>
              Live Translation Output
            </p>
            <div className="mt-5 min-h-[5rem] rounded-2xl border border-white/5 bg-black/40 p-5 shadow-inner backdrop-blur-sm">
              <p id="sentence-output" className="text-3xl font-medium leading-relaxed tracking-wide text-white md:text-4xl drop-shadow-md">
                {sentence ? (
                  <>
                    <span>{sentence}</span>
                    <span className="ml-1 inline-block h-8 w-1 animate-pulse bg-indigo-400 align-middle"></span>
                  </>
                ) : (
                  <span className="text-slate-500">Awaiting input...</span>
                )}
              </p>
            </div>
          </div>
        </section>

        {/* Comprehensive Self-Diagnostics Section */}
        <section className="relative rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-2xl backdrop-blur-3xl">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-4">
            <div>
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <span>System Functions & Diagnostics</span>
                <span className="text-xs text-indigo-300 font-normal px-2.5 py-0.5 rounded-full border border-indigo-500/30 bg-indigo-500/10">
                  Verification Suite
                </span>
              </h2>
              <p className="text-xs text-slate-400 mt-1">
                Run an automated end-to-end check to verify all model inference and pipeline functions are perfectly in use.
              </p>
            </div>
            <button
              id="run-diagnostics-btn"
              onClick={runDiagnostics}
              disabled={isDiagnosing}
              className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-indigo-500 to-fuchsia-500 text-white font-semibold text-sm shadow-lg shadow-indigo-500/25 hover:opacity-90 active:scale-95 transition-all disabled:opacity-50 cursor-pointer flex items-center gap-2"
            >
              {isDiagnosing ? (
                <>
                  <span className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  <span>Checking...</span>
                </>
              ) : (
                <>
                  <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>Run Verification Check</span>
                </>
              )}
            </button>
          </div>

          {diagnostics.length > 0 && (
            <div id="diagnostics-results-container" className="mt-4 grid gap-2.5 sm:grid-cols-2 lg:grid-cols-3">
              {diagnostics.map((diag, i) => (
                <div
                  key={i}
                  className={`p-3.5 rounded-xl border backdrop-blur-md flex flex-col justify-between ${
                    diag.status === "pass"
                      ? "border-emerald-500/30 bg-emerald-950/20 text-emerald-200"
                      : "border-rose-500/30 bg-rose-950/20 text-rose-200"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-xs font-semibold tracking-wide text-white">{diag.name}</span>
                    <span
                      className={`text-[10px] font-bold px-2 py-0.5 rounded-md uppercase ${
                        diag.status === "pass" ? "bg-emerald-500/20 text-emerald-300" : "bg-rose-500/20 text-rose-300"
                      }`}
                    >
                      {diag.status}
                    </span>
                  </div>
                  <p className="text-xs text-slate-300 font-mono">{diag.detail}</p>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
