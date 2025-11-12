# 🌿 Crop Disease Identifier using FastAI (Local Dataset)
# Author: Vinod
# Description: Trains a CNN on tomato leaf dataset downloaded from Kaggle

from fastai.vision.all import *
import pathlib

# --- Fix Windows Path issues ---
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# --- Path to your local dataset ---
# ✅ Change this path if your dataset is stored elsewhere
path = Path(r"C:\Users\Vinod\OneDrive\Desktop\crop_disease_app\data\tomato")

# --- Verify dataset folders ---
if not path.exists():
    raise FileNotFoundError(f"❌ Dataset folder not found at: {path}")

classes = [f.name for f in path.iterdir() if f.is_dir()]
if not classes:
    raise ValueError("❌ No class folders found inside dataset path! Please check structure.")

print(f"📂 Found classes: {classes}")

# --- Step 1: Create DataLoaders ---
dls = ImageDataLoaders.from_folder(
    path,
    valid_pct=0.2,
    seed=42,
    item_tfms=Resize(224),
    batch_tfms=aug_transforms(),
    bs=16,
    num_workers=0
)

# --- Step 2: Show sample images ---
print("\n📸 Sample training images:")
dls.show_batch(max_n=9, figsize=(6, 6))

# --- Step 3: Define and Train CNN ---
learn = vision_learner(dls, resnet34, metrics=accuracy)
print("\n🚀 Training model...")
learn.fine_tune(3)
learn.export("tomato_disease_model.pkl")
print("\n✅ Model training complete! Saved as 'tomato_disease_model.pkl'")

# --- Step 4: Test Prediction on a Sample Image ---
# Pick a random image file from the first class folder
import random

first_class = classes[0]
image_files = get_image_files(path/first_class)
sample_img = random.choice(image_files)

print(f"\n🧪 Predicting disease for sample image: {sample_img}")
img = PILImage.create(sample_img)
pred, pred_idx, probs = learn.predict(img)


# --- Step 5: Suggest Possible Cure ---
disease_cures = {
    "Tomato___Bacterial_spot": "Use copper-based fungicide and avoid overhead watering.",
    "Tomato___Leaf_Mold": "Improve air circulation and apply chlorothalonil-based fungicide.",
    "Tomato___healthy": "No disease detected. Maintain regular monitoring."
}

print("💊 Suggested Cure:", disease_cures.get(pred, "No information available."))
