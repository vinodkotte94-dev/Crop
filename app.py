from fastai.vision.all import *
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="🌿 Crop Disease Identifier", layout="wide")

MODEL_PATH = Path("model.pkl")      # ✅ Correct file

@st.cache_resource
def load_model():
    try:
        learn = load_learner(MODEL_PATH, cpu=True)   # ✅ Native FastAI load
        return learn
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

learn = load_model()
st.title("🌿 Prompt-Based Crop Disease Identifier")
st.markdown("### Identify plant leaf diseases using a FastAI trained model")

def get_cure_suggestion(disease_name: str):
    disease = disease_name.lower()

    cures = {
        "healthy": "No treatment required.",
        "bacterial spot": "- Apply copper fungicides.\n- Avoid overhead watering.",
        "leaf mold": "- Use sulfur fungicides.\n- Improve ventilation.",
        "early blight": "- Apply mancozeb/chlorothalonil.\n- Remove infected leaves.",
        "late blight": "- Apply copper sulfate.\n- Destroy affected plants.",
        "septoria": "- Use mancozeb/chlorothalonil.\n- Prune lower leaves.",
    }
    for key in cures:
        if key in disease:
            return cures[key]
    return "⚠ Unknown disease."

prompt = st.text_input("💬 Enter your prompt")
uploaded = st.file_uploader("📸 Upload leaf image", type=["jpg","jpeg","png"])

if uploaded and prompt:
    img = PILImage.create(uploaded)
    st.image(img.to_thumb(400,400))

    with st.spinner("Analyzing..."):
        pred_class, pred_idx, probs = learn.predict(img)

    clean = pred_class.replace("_"," ").title()

    st.success(f"Disease: {clean}")
    st.info(f"Confidence: {probs[pred_idx]:.2%}")
    st.markdown(get_cure_suggestion(clean))

else:
    st.warning("Enter a prompt and upload an image to continue.")
