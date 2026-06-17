# Hardware Setup Guide

Step-by-step instructions to set up either the Jetson Nano or Raspberry Pi 4 for this project.

---

## Option A — Jetson Nano 4GB (Recommended)

### What you need
- Jetson Nano Developer Kit (4GB)
- 32GB+ microSD card (A2 speed class)
- 5V 4A DC barrel jack power supply (**not** micro-USB — it cannot supply enough current)
- IMX219 CSI camera module OR USB webcam (Logitech C270 / C920)
- Micro-USB to USB-A cable (for serial access during setup)
- Ethernet cable
- A heatsink + small 5V fan (required — the Nano throttles under inference load without cooling)

### Step 1 — Flash JetPack

1. Download **JetPack 4.6.4** from the NVIDIA Developer site:  
   `https://developer.nvidia.com/embedded/jetpack-sdk-464`
2. Flash to microSD using **balenaEtcher**
3. Insert SD card, connect ethernet, power on with the barrel jack
4. Complete the first-boot setup (set username, password, timezone)

### Step 2 — Check CUDA and TensorRT

```bash
# Verify CUDA
nvcc --version

# Verify TensorRT
python3 -c "import tensorrt; print('TRT version:', tensorrt.__version__)"

# Verify OpenCV with CUDA
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i cuda
```

Expected TensorRT version: `8.2.x`

### Step 3 — Install PyTorch for Jetson

NVIDIA provides custom PyTorch wheels for JetPack. Do NOT use `pip install torch` — it won't use CUDA.

```bash
# For JetPack 4.6, Python 3.8
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl \
     -O torch-1.11.0-cp38-cp38-linux_aarch64.whl

pip3 install torch-1.11.0-cp38-cp38-linux_aarch64.whl

# Verify
python3 -c "import torch; print(torch.cuda.is_available())"  # should print True
```

### Step 4 — Install project dependencies

```bash
git clone https://github.com/YOUR_USERNAME/traffic-density-estimator.git
cd traffic-density-estimator
pip3 install -r requirements_jetson.txt
```

### Step 5 — Camera setup

**USB webcam:**
```bash
ls /dev/video*   # should show /dev/video0
python3 src/main.py --source 0
```

**IMX219 CSI camera (V4L2 mode):**
```bash
# Enable V4L2 driver
sudo modprobe nvhost_vi
sudo modprobe nvhost_isp

# Test
gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! autovideosink

# Run project
python3 src/main.py --source "nvarguscamerasrc ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink"
```

### Step 6 — Export to TensorRT

```bash
python3 scripts/download_model.py
python3 scripts/export_tensorrt.py --precision fp16

# Run with TRT engine
python3 src/main.py --source 0 --trt-engine models/yolov8n_fp16.engine
```

### Step 7 — Benchmark

```bash
python3 scripts/benchmark.py \
    --trt-engine models/yolov8n_fp16.engine \
    --iterations 200
```

---

## Option B — Raspberry Pi 4 (Budget)

### What you need
- Raspberry Pi 4 Model B (4GB or 8GB RAM)
- 32GB+ microSD (A2 rated)
- Official 5V 3A USB-C power supply
- Pi Camera Module v2 OR USB webcam
- Ethernet cable

### Step 1 — Flash Raspberry Pi OS

1. Use **Raspberry Pi Imager** to flash **Raspberry Pi OS (64-bit, Lite or Desktop)** to microSD
2. In Imager settings: enable SSH, set hostname, username, password, WiFi (optional)
3. Boot and connect via SSH: `ssh pi@raspberrypi.local`

### Step 2 — System setup

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv libopenblas-dev libatlas-base-dev \
    libhdf5-dev libopencv-dev python3-opencv git

python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install ONNX Runtime (no GPU)

```bash
pip install onnxruntime   # CPU version
```

### Step 4 — Install project dependencies

```bash
git clone https://github.com/YOUR_USERNAME/traffic-density-estimator.git
cd traffic-density-estimator
pip install -r requirements.txt
```

### Step 5 — Download and export model to ONNX

```bash
# Download weights (on a more powerful PC, then copy to Pi)
python scripts/download_model.py --onnx

# Copy the .onnx file to models/ on the Pi via scp:
scp models/yolov8n.onnx pi@raspberrypi.local:/home/pi/traffic-density-estimator/models/
```

### Step 6 — Run (ONNX, no TRT)

```bash
python src/main.py --source 0 --no-trt --model models/yolov8n.onnx
```

Expected performance: ~4–6 FPS at 640px. Reduce `--img-size 320` for faster inference.

---

## Performance Tuning Tips

### Jetson Nano
```bash
# Max performance mode (runs CPU + GPU at max clock)
sudo nvpmodel -m 0
sudo jetson_clocks

# Monitor GPU usage
sudo tegrastats
```

### Raspberry Pi 4
```bash
# Reduce input resolution for higher FPS
python src/main.py --source 0 --img-size 320 --no-trt

# Monitor CPU temperature
vcgencmd measure_temp
```

---

## Troubleshooting

| Problem | Solution |
|---------|---------|
| `Cannot open source: 0` | Try `--source 1` or `--source /dev/video0` |
| TRT export OOM on Jetson | Reduce `--workspace 1` |
| Low FPS on Pi | Use `--img-size 320` and `--no-trt` |
| Dashboard not loading | Open `http://<device-ip>:8000` — not `localhost` if on another machine |
| `import tensorrt` fails | Only works with JetPack 4.6 and NVIDIA GPU; use ONNX on Pi |
