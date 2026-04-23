# 🌿 FreshHarvest AI Inspector

An AI-powered fruit freshness classification system built for **FreshHarvest Logistics** a California-based produce distributor. The system automates quality inspection on warehouse conveyor belts by classifying fruit images as **fresh** or **spoiled** in real time.

---

## Demo

Upload a fruit image and the model predicts its freshness instantly with confidence scores and a full top-5 breakdown.

---

## Supported Fruits

| Fruit | Fresh Class | Spoiled Class |
|---|---|---|
| 🍌 Banana | `F_Banana` | `S_Banana` |
| 🍋 Lemon | `F_Lemon` | `S_Lemon` |
| 🟡 Lulo | `F_Lulo` | `S_Lulo` |
| 🥭 Mango | `F_Mango` | `S_Mango` |
| 🍊 Orange | `F_Orange` | `S_Orange` |
| 🍓 Strawberry | `F_Strawberry` | `S_Strawberry` |
| 🔴 Tamarillo | `F_Tamarillo` | `S_Tamarillo` |
| 🍅 Tomato | `F_Tomato` | `S_Tomato` |

**16 total classes** — 8 fruits × 2 freshness states.

---

## Project Structure

```
freshharvest/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── freshharvest_resnet50_final.pth # Trained model checkpoint (not in repo see below)
│
└── notebooks/
    └── FreshHarvest_Training.ipynb # Full training pipeline
```

---

## Model Architecture

| Component | Detail |
|---|---|
| **Base model** | ResNet50 (ImageNet pretrained) |
| **Strategy** | 3-phase transfer learning |
| **Input size** | 128 × 128 pixels |
| **Classifier head** | Linear(2048→512) → BN → ReLU → Dropout(0.4) → Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→16) |
| **Loss function** | Weighted CrossEntropy (1.8× boost on hard classes) |
| **Test accuracy** | ~98% on held-out test set |

### Training Phases

| Phase | Layers trained | Learning rate |
|---|---|---|
| Phase 1 | Classifier head only | `1e-3` |
| Phase 2 | ResNet layer4 + head | backbone `1e-5`, head `2e-4` |
| Phase 3 | ResNet layer3 + layer4 + head | layer3 `5e-6`, layer4 `1e-5`, head `1e-4` |

---

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/freshharvest-ai.git
cd freshharvest-ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the model checkpoint

The model file (`freshharvest_resnet50_final.pth`) is too large for GitHub. Download it and place it in the root folder:

```
freshharvest-ai/
└── freshharvest_resnet50_final.pth   ← place here
```

> If you want to train from scratch, open `notebooks/FreshHarvest_Training.ipynb` and follow the cells in order.

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## Dataset

The model was trained on a hybrid dataset (**FRUIT16K_v2**) combining:

| Source | Classes | Images per class |
|---|---|---|
| Kaggle — Fruits & Vegetables Dataset | Banana, Mango, Orange, Strawberry, Tomato (fresh + spoiled) | ~600 |
| FRUIT16K (original) | Lemon, Lulo, Tamarillo (fresh + spoiled) | ~600 |

**Total: ~9,600 images across 16 classes.**

### Data augmentation (training)
- Random horizontal & vertical flip
- Random rotation (±40°)
- Colour jitter (brightness, contrast, saturation, hue)
- Random grayscale
- Random perspective
- Random erasing (applied after ToTensor)
- ImageNet normalisation

---

## App Features

| Feature | Detail |
|---|---|
| **Drag & drop upload** | Drop one or multiple images at once |
| **Multi-image** | No page refresh needed between images |
| **Background removal** | `rembg` strips background before prediction (reduces colour bias) |
| **Test-time augmentation** | 6 augmented views averaged for robustness |
| **Confidence bar** | Visual indicator of prediction certainty |
| **Top-5 breakdown** | Full ranked prediction list |
| **Low confidence warning** | Flags results below 65% for manual review |
| **Processed view** | Toggle to see background-removed image |

---

## Known Limitations

| Class | Limitation |
|---|---|
| **Lulo** | Visually similar to unripe mango; low diversity in training data |
| **Tamarillo** | Resembles tomato at certain angles |
| **Lemon** | Small dataset from original FRUIT16K; may be confused with orange |
| **All classes** | Cut/sliced fruit not in training data whole fruit only |

For best results:
- Use a **single fruit** that fills most of the frame
- **Plain white background** (or let background removal handle it)
- **Good even lighting**, no harsh shadows
- **Whole fruit only**, not cut or peeled

---

## Tech Stack

| Component | Technology |
|---|---|
| Web app | Streamlit |
| Deep learning | PyTorch |
| Model | ResNet50 (torchvision) |
| Background removal | rembg |
| Image processing | Pillow, torchvision.transforms |
| Data analysis | NumPy, scikit-learn, seaborn |

---

## Training the Model Yourself

Open `notebooks/FreshHarvest_Training.ipynb` in Jupyter and run cells in order:

```
Cell 1   → Imports & seed
Cell 2   → Config (set DATA_DIR here)
Cell 3   → Transforms
Cell 4   → Dataset loader
Cell 5   → Class-weighted loss
Cell 6   → Build ResNet50
Cell 7   → Training utilities
Cell 8   → Phase 1 training
Cell 9   → Phase 2 fine-tuning
Cell 10  → Phase 3 fine-tuning
Cell 11  → Training curves
Cell 12  → Test evaluation + confusion matrix
Cell 13  → Save model
```

Expected training time on CPU: ~2–3 hours total across all 3 phases.
On GPU: ~15–25 minutes.

---

## Acknowledgements

- [CodeBasics](https://codebasics.io/bootcamps/dashboard/gen-ai-data-science-bootcamp-with-virtual-internship) — Part of virtual internship
- [ResNet50](https://arxiv.org/abs/1512.03385) — He et al., 2016
- [Kaggle Fruits & Vegetables Dataset](https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset)
- [rembg](https://github.com/danielgatis/rembg) — background removal
- [Streamlit](https://streamlit.io) — web app framework
