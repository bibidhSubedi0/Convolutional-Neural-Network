# Convolutional Neural Network — From Scratch in C++

A handwritten digit classifier built entirely from scratch in C++, with no ML libraries. Implements a full CNN pipeline including forward propagation, backpropagation through convolutional layers, max pooling with gradient unpooling, and a fully connected network with softmax output — all using custom matrix and layer implementations.

A browser-based visualizer lets you draw a digit and watch activations propagate through every layer in real time.

![Architecture](https://img.shields.io/badge/architecture-CNN-00ffe0?style=flat-square) ![Language](https://img.shields.io/badge/language-C%2B%2B17-blue?style=flat-square) ![Dataset](https://img.shields.io/badge/dataset-MNIST-ff4f7b?style=flat-square)

---
## Visualization
<img width="1906" height="915" alt="image" src="https://github.com/user-attachments/assets/af742c0f-7539-450f-83c9-eb9c68f53b8a" />

---
## Architecture

```
Input (28×28)
    │
    ▼
Conv Layer 1 — 3 hardcoded 5×5 edge detection filters → 3 × (24×24) feature maps
    │  ReLU
    ▼
Max Pool 1 — 2×2 stride 2 → 3 × (12×12)
    │
    ▼
Conv Layer 2 — 3 trainable 3×3×3 filters → 3 × (10×10) feature maps
    │  ReLU
    ▼
Max Pool 2 — 2×2 stride 2 → 3 × (5×5) → flatten → 75
    │
    ▼
Fully Connected — 75 → 256 (Leaky ReLU)
    │
    ▼
Output — 256 → 10 (Softmax)
```

**Conv Layer 1** uses fixed edge detection filters (vertical, horizontal, diagonal). **Conv Layer 2** uses Xavier-initialized filters that are trained via backpropagation. The FC network uses cross-entropy loss with combined softmax + cross-entropy gradient for numerical stability.

---

## Features

- **No ML framework** — every operation (convolution, pooling, backprop, weight updates) written from scratch
- **Full backpropagation** through conv layer 2, including max-pool gradient unpooling and ReLU derivative masking
- **Gradient clipping** in both conv and FC layers to stabilize training
- **Model serialization** — saves/loads all trained weights and filters to a binary file
- **Learning rate decay** — conv learning rate halved every 5 epochs
- **Live visualizer** — HTML frontend shows feature maps, FC activations, and output probabilities as you draw


---

## Dependencies

- **C++17** compiler (MSVC, g++, or clang++)
- **OpenCV** — image loading only (`cv::imread`)
- **Python 3** + `flask flask-cors pillow numpy` — for the visualizer server only

---

## Build

### Visual Studio (Windows)
Open the solution, set the include/library paths for OpenCV, build in Debug or Release.

### g++ (Linux/macOS)
```bash
g++ -std=c++17 -O2 main.cpp cnn/*.cpp -o cnn \
    $(pkg-config --cflags --libs opencv4)
```

---

## Usage

### Train
```bash
./cnn train [model_path]
# default: saves to saved_model.bin
# trains for 15 epochs on mnist_train.csv
```

### Evaluate
```bash
./cnn eval [model_path]
# evaluates on mnist_test.csv (first 1000 samples)
```

### Predict a single image
```bash
./cnn predict image.jpg [model_path]
```
> Image must be 28×28 grayscale, white digit on black background (MNIST style).

---

## Visualizer

A browser UI that lets you draw a digit and watch it propagate through the network layer by layer.

**Start the server:**
```bash
pip install flask flask-cors pillow numpy
python visualizer/server.py
```
Then open `visualizer/index.html` in your browser.

The visualizer shows real conv layer 1 activations (exact match to C++ pipeline), approximated conv layer 2 maps, a 16×16 heatmap of FC neuron activations, and real model probabilities from the C++ executable.

If the server isn't running, the page falls back to a demo mode that runs the edge filter math entirely in JavaScript.

---

## Implementation Notes

**Why two conv layers with different approaches?**  
Layer 1 uses fixed classical edge filters — these are known to work well for digit recognition and don't need training. Layer 2 learns task-specific features from the pooled edge maps.

**Backprop through max pooling**  
Uses "unpool without indices" — gradients are routed back to the position of the max value in the original feature map, which is recomputed during the backward pass.

**Softmax + cross-entropy gradient**  
The combined gradient simplifies to `(predicted - target)`, computed directly in `gardientComputation()` to avoid numerical issues from separately differentiating softmax and log-loss.

**Gradient clipping**  
Both the FC weight updates and conv filter updates clip gradients to `[-1, 1]` and `[-1, 1]` respectively, which significantly stabilized training on small datasets.

---

## Results

Trained on 5,000 MNIST samples for 15 epochs:

| Metric | Value |
|---|---|
| Training samples | 5,000 |
| Epochs | 15 |
| FC learning rate | 0.001 |
| Conv learning rate | 0.00001 (halved every 5 epochs) |

> Training on the full 60,000-sample dataset is supported — change the `maxSamples` argument in `runTraining()`.

---

## License

MIT
