# 🌿 Improved Prompt-Based Crop Disease Identifier using FastAI + Streamlit

from fastai.vision.all import *
import streamlit as st
from pathlib import Path

MODEL_PATH = Path("model_clean.pkl")

# ---------------------------------------------------------
# Load model safely for Linux (Streamlit Cloud)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        learn = load_learner(MODEL_PATH, cpu=True)
        return learn
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

def get_cure_suggestion(disease_name: str):
    disease = disease_name.lower()

    cures = {
        "healthy": "The leaf is healthy. No treatment required.",
        "bacterial spot": (
            "- Apply copper-based fungicides.\n"
            "- Avoid overhead watering."
        ),
        "leaf mold": (
            "- Use sulfur fungicides.\n"
            "- Increase air ventilation."
        ),
        "early blight": (
            "- Use mancozeb/chlorothalonil.\n"
            "- Remove infected leaves."
        ),
        "late blight": (
            "- Apply copper sulfate.\n"
            "- Destroy infected plants immediately."
        ),
        "septoria leaf spot": (
            "- Use chlorothalonil or mancozeb fungicides.\n"
            "- Prune lower leaves."
        ),
    }

    for key in cures:
        if key in disease:
            return cures[key]

    return "⚠ Unknown disease. Dataset label may be incorrect."

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="🌾 Crop Disease Identifier", layout="wide")

st.title("🌿 Prompt-Based Crop Disease Identifier")
st.markdown("### Identify plant leaf diseases using a FastAI Trained Model")

learn = load_model()   # ✅ FIX — safe loading

prompt = st.text_input("💬 Enter your prompt")
uploaded = st.file_uploader("📸 Upload leaf image", type=["jpg", "jpeg", "png"])

if uploaded and prompt:
    img = PILImage.create(uploaded)
    st.image(img.to_thumb(400, 400), caption="Uploaded Image")

    with st.spinner("🔍 Analyzing..."):
        pred_class, pred_idx, probs = learn.predict(img)

    clean = str(pred_class).replace("_", " ").title()

    st.subheader("🩺 Prediction")
    st.success(f"**Disease:** {clean}")
    st.info(f"**Confidence:** {probs[pred_idx]:.2%}")

    st.subheader("💊 Cure Suggestion")
    st.markdown(get_cure_suggestion(clean))

else:
    st.warning("💡 Enter a prompt and upload an image to continue.")
