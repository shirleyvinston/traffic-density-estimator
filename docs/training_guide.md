# Training on Custom Traffic Data

How to collect, annotate, and fine-tune YOLOv8n on your own traffic footage — the step that makes this project truly yours.

---

## Why custom training?

YOLOv8n is pretrained on COCO (a general dataset). Custom training on local traffic footage gives:
- Higher accuracy for your specific camera angle and scene
- Better handling of local vehicle types (auto-rickshaws, two-wheelers, etc.)
- A model you can claim full ownership of in interviews

This is also the step most students skip — which means doing it sets you apart.

---

## Step 1 — Collect footage

Record 5–15 minutes of traffic from a fixed camera angle. Tips:
- Shoot at the same height and angle you'll use in deployment
- Include varied lighting: day, dusk, night if possible
- Vary traffic density: light, moderate, heavy
- Aim for 720p or 1080p

Useful public datasets to supplement:
- **UA-DETRAC**: `https://detrac-db.rit.albany.edu`
- **MIO-TCD**: `https://tcd.miovision.com`
- **COCO vehicle subset**: downloadable via Roboflow Universe

---

## Step 2 — Annotate with Roboflow

1. Create a free account at `https://roboflow.com`
2. Create a new project → **Object Detection**
3. Upload your video — Roboflow extracts frames automatically
4. Annotate bounding boxes for: `car`, `truck`, `bus`, `motorcycle`, `bicycle`
5. Enable **Smart Polygon** (auto-suggest boxes using SAM)
6. Apply augmentations:
   - Flip: horizontal only
   - Brightness: ±25%
   - Blur: up to 1px
   - Noise: up to 2%
7. Export → **YOLOv8 format** → copy your API key

Aim for at least **200 annotated frames** (more = better).

---

## Step 3 — Train

```bash
python scripts/train_custom.py \
    --roboflow-key YOUR_API_KEY \
    --workspace YOUR_WORKSPACE \
    --project YOUR_PROJECT_NAME \
    --version 1 \
    --epochs 50 \
    --batch 16
```

For Jetson Nano (limited RAM, use smaller batch):
```bash
python scripts/train_custom.py \
    --roboflow-key YOUR_KEY \
    --workspace ws --project proj --version 1 \
    --epochs 50 --batch 4
```

Training takes ~30 min on a modern GPU, ~4–6 hours on Jetson Nano.

---

## Step 4 — Monitor with TensorBoard

```bash
tensorboard --logdir runs/detect/
```

Open `http://localhost:6006` in your browser. Watch:
- `train/box_loss` — should decrease steadily
- `val/mAP50` — should rise above 0.70 for a decent model
- `val/mAP50-95` — harder metric; aim for 0.45+

---

## Step 5 — Evaluate

```bash
python scripts/train_custom.py \
    --validate-only runs/detect/traffic_custom/weights/best.pt \
    --data data/roboflow/data.yaml
```

Expected results on a well-annotated local dataset:
- mAP50 > 0.75
- mAP50-95 > 0.45

---

## Step 6 — Deploy the custom model

```bash
# Export to TensorRT (Jetson)
python scripts/export_tensorrt.py \
    --model runs/detect/traffic_custom/weights/best.pt \
    --precision fp16

# Run with custom engine
python src/main.py \
    --source 0 \
    --trt-engine runs/detect/traffic_custom/weights/best_fp16.engine
```

---

## Interview talking points from this step

- "I annotated X frames using Roboflow with smart polygon assistance, then applied mosaic and mixup augmentation during training"
- "Fine-tuning improved mAP50 from 0.68 (COCO pretrain) to 0.81 on our local traffic scene"
- "I used early stopping with patience=20 to avoid overfitting on the small dataset"
