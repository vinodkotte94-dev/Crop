# 🧠 Tomato Leaf Disease Identifier using FastAI
# ---------------------------------------------
# ✅ Works with local Kaggle dataset
# ✅ Adds detailed cure suggestions

from fastai.vision.all import *
from pathlib import Path
import matplotlib.pyplot as plt

def train_and_predict():
    # Step 1: Path to your local dataset
    path_data = Path(r"C:\Users\Vinod\OneDrive\Desktop\crop_disease_app\data\tomato")

    if not path_data.exists():
        raise FileNotFoundError(f"❌ Dataset not found at: {path_data}")
    print(f"📂 Using dataset from: {path_data}")

    # Step 2: Create DataLoaders
    dls = ImageDataLoaders.from_folder(
        path_data,
        valid_pct=0.2,
        seed=42,
        item_tfms=Resize(224),
        num_workers=0
    )

    # Step 3: Train the model
    print("\n🚀 Training model (please wait)...")
    learn = vision_learner(dls, resnet18, metrics=accuracy)
    learn.fine_tune(3)

    # Step 4: Save the trained model
    model_path = Path(r"C:\Users\Vinod\OneDrive\Desktop\crop_disease_app\tomato_disease_model.pkl")
    learn.export(model_path)
    print(f"\n✅ Model saved successfully at: {model_path}")

    # Step 5: Test a prediction
    sample_img = get_image_files(path_data)[0]
    img = PILImage.create(sample_img)
    pred_class, pred_idx, probs = learn.predict(img)

    # Clean up prediction name
    clean_name = str(pred_class).replace("Tomato___", "").replace("_", " ")

    print(f"\n🪴 Predicted Disease: {clean_name}")
    print(f"📊 Confidence: {probs[pred_idx]:.2%}")

    # Step 6: Suggest treatment
    print("\n💊 Treatment Suggestion:")
    disease = clean_name.lower()

    if "healthy" in disease:
        print("✅ The leaf is healthy. No treatment required.")
    elif "bacterial spot" in disease:
        print("⚠️ Disease: Bacterial Spot Detected!")
        print("👉 Apply copper-based fungicides like copper hydroxide or copper oxychloride.")
        print("👉 Avoid overhead watering and ensure good air circulation.")
        print("👉 Remove infected leaves to prevent spread.")
    elif "leaf mold" in disease:
        print("⚠️ Disease: Leaf Mold Detected!")
        print("👉 Improve air ventilation and reduce humidity.")
        print("👉 Use sulfur-based or chlorothalonil fungicide.")
        print("👉 Avoid watering late in the day.")
    elif "early blight" in disease:
        print("⚠️ Disease: Early Blight Detected!")
        print("👉 Use fungicides containing mancozeb or chlorothalonil.")
        print("👉 Rotate crops and remove infected debris.")
    elif "late blight" in disease:
        print("⚠️ Disease: Late Blight Detected!")
        print("👉 Apply fungicides like copper sulfate or metalaxyl.")
        print("👉 Destroy affected plants to prevent spread.")
    elif "septoria" in disease:
        print("⚠️ Disease: Septoria Leaf Spot Detected!")
        print("👉 Use fungicides with chlorothalonil or mancozeb.")
        print("👉 Prune lower leaves and improve spacing.")
    else:
        print("⚠️ Unknown disease type. Please verify your dataset folder names.")

    # Step 7: Show the image
    img.show(title=f"Prediction: {clean_name}")
    plt.show()

# ✅ Windows safe entry point
if __name__ == "__main__":
    train_and_predict()
