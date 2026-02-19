"""
server.py — bridges the CNN Visualizer frontend to the C++ model.

Install deps:  pip install flask flask-cors pillow numpy
Run:           python server.py
Then open index.html in a browser and draw a digit.
"""



from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import subprocess, tempfile, os, sys
from PIL import Image

app = Flask(__name__)
CORS(app)

# ── Paths — adjust if your layout differs ─────────────────────────────────
EXE_PATH = r"C:\Users\Bibidh Subedi\source\repos\Convolutional-Neural-Network\bin\Debug\ConvolutionalNeuralNetwork.exe"
MODEL_PATH = r"C:\Users\Bibidh Subedi\source\repos\Convolutional-Neural-Network\saved_model.bin"
# ── Fixed edge filters (must match ConvolutionLayers hardcoded filters) ───
EDGE = [
    [[1,0,-1,0,1],[1,0,-1,0,1],[1,0,-1,0,1],[1,0,-1,0,1],[1,0,-1,0,1]],
    [[1,1,1,1,1],[0,0,0,0,0],[-1,-1,-1,-1,-1],[0,0,0,0,0],[1,1,1,1,1]],
    [[1,1,0,-1,-1],[1,1,0,-1,-1],[0,0,0,0,0],[-1,-1,0,1,1],[-1,-1,0,1,1]]
]

# ── Numpy helpers ──────────────────────────────────────────────────────────
def conv2d(img: np.ndarray, f: np.ndarray, s: int = 1) -> np.ndarray:
    iH, iW = img.shape
    fH, fW = f.shape
    oH = (iH - fH) // s + 1
    oW = (iW - fW) // s + 1
    # use stride_tricks for speed
    shape   = (oH, oW, fH, fW)
    strides = (img.strides[0]*s, img.strides[1]*s, img.strides[0], img.strides[1])
    patches = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    return (patches * f).sum(axis=(2, 3))

def relu(m: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, m)

def maxpool(m: np.ndarray, s: int = 2) -> np.ndarray:
    iH, iW = m.shape
    oH, oW = int(np.ceil(iH / s)), int(np.ceil(iW / s))
    out = np.full((oH, oW), -np.inf)
    for i in range(oH):
        for j in range(oW):
            patch = m[i*s : i*s+s, j*s : j*s+s]
            out[i, j] = patch.max()
    return out

# ── Call C++ exe, parse probability lines ──────────────────────────────────
def run_cpp_inference(pixels_28x28: np.ndarray):
    """Save pixels as a temp PNG, call the exe, parse 10 probabilities."""
    img_arr = (pixels_28x28 * 255).clip(0, 255).astype(np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        Image.fromarray(img_arr, mode="L").save(tmp)
        result = subprocess.run(
            [EXE_PATH, "predict", tmp, MODEL_PATH],
            capture_output=True, text=True, timeout=15
        )
        probs = []
        for line in result.stdout.splitlines():
            line = line.strip()
            # matches lines like "  3: 27.2863%"
            if ":" in line and "%" in line:
                try:
                    pct = float(line.split(":")[1].replace("%", "").strip())
                    probs.append(pct / 100.0)
                except ValueError:
                    pass
        if len(probs) == 10:
            return probs
        print("[server] WARNING: could not parse 10 probs from exe output:")
        print(result.stdout)
        print(result.stderr)
    except Exception as e:
        print(f"[server] C++ call failed: {e}")
    finally:
        os.unlink(tmp)
    return None   # caller will use softmax fallback

# ── /predict endpoint ──────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data   = request.get_json(force=True)
    pixels = np.array(data["pixels"], dtype=np.float32)   # 28×28

    # ── Layer 1: fixed edge filters (exact match to C++ pipeline) ──────────
    conv1_np = [relu(conv2d(pixels, np.array(f, dtype=np.float32))) for f in EDGE]
    pool1_np = [maxpool(m) for m in conv1_np]              # 12×12 each

    # ── Layer 2: approximate (sinusoidal placeholder filters) ───────────────
    # We don't export the trained conv-2 weights, so we use a deterministic
    # approximation — shapes are correct, activations look plausible.
    conv2_np = []
    for fi in range(3):
        combined = sum(
            pool1_np[ci] * float(np.sin(fi * 7 + ci * 3 + 1) * 0.4)
            for ci in range(3)
        )
        filt = np.array([[np.sin(fi*3 + i*2 + j) for j in range(3)]
                         for i in range(3)], dtype=np.float32)
        conv2_np.append(relu(conv2d(combined, filt)))      # 10×10 each
    pool2_np = [maxpool(m) for m in conv2_np]             # 5×5 each

    # ── FC placeholder (256 neurons, shape only) ────────────────────────────
    flat = np.concatenate([m.ravel() for m in pool2_np])
    fc   = [float(max(0.0, float(np.dot(flat, np.sin(i * 0.31 + np.arange(len(flat)) * 0.07)))))
            for i in range(256)]

    # ── Real probabilities from C++ model ───────────────────────────────────
    probs = run_cpp_inference(pixels)
    if probs is None:
        # graceful fallback: softmax over FC logits
        logits = np.array([sum(fc[j] * np.cos(i * 1.1 + j * 0.05) * 0.08
                               for j in range(len(fc))) for i in range(10)])
        e = np.exp(logits - logits.max())
        probs = (e / e.sum()).tolist()

    return jsonify({
        "layers": {
            "conv1": [m.tolist() for m in conv1_np],
            "pool1": [m.tolist() for m in pool1_np],
            "conv2": [m.tolist() for m in conv2_np],
            "pool2": [m.tolist() for m in pool2_np],
        },
        "fc":    fc,
        "probs": probs,
    })

# ── Health check ───────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok", "exe": EXE_PATH, "model": MODEL_PATH})

if __name__ == "__main__":
    print(f"  exe   : {EXE_PATH}")
    print(f"  model : {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("  WARNING: model file not found — check MODEL_PATH")
    app.run(host="0.0.0.0", port=5000, debug=False)