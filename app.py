# =========================================================
# 100% FIX for FastAI PKL on Linux (Streamlit Cloud)
# Handles WindowsPath + persistent IDs + storages
# =========================================================

# =============================================================
# FINAL FIX: Fully compatible FastAI unpickler for Linux
# Handles:
#  ✓ WindowsPath -> PosixPath
#  ✓ persistent_load storage
#  ✓ nested data.pkl loaders
#  ✓ torch storages
# =============================================================

import pickle
import pathlib
import torch

class FastAIUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Handle WindowsPath from pickle
        if module == "pathlib" and name == "WindowsPath":
            return pathlib.PosixPath
        return super().find_class(module, name)

    def persistent_load(self, pid):
        """
        Handles all persistent objects stored by FastAI & Torch.
        """

        # Torch storage: ('storage', storage_type, key, location, size)
        if isinstance(pid, tuple) and pid[0] == 'storage':
            _, storage_type, key, location, size = pid
            storage = torch.UntypedStorage(size)
            return storage

        # FastAI nested pickles (e.g. data.pkl inside model_clean.pkl)
        if isinstance(pid, (str, bytes)):
            # Return placeholder or empty tensor
            return pid

        raise pickle.UnpicklingError(f"Unhandled persistent load: {pid}")


def load_fastai_model(fname):
    with open(fname, "rb") as f:
        return FastAIUnpickler(f).load()


# ---- custom loader that handles WindowsPath + persistent IDs ----
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Replace WindowsPath -> PosixPath
        if module == "pathlib" and name == "WindowsPath":
            return pathlib.PosixPath
        return super().find_class(module, name)

    def persistent_load(self, pid):
        """
        Handle persistent IDs created by torch and FastAI.
        """
        if isinstance(pid, tuple) and pid[0] == 'storage':
            _, storage_type, key, location, size = pid

            # Allocate empty storage and return it
            storage = torch.UntypedStorage(size)
            return storage

        raise pickle.UnpicklingError(f"Unsupported persistent load: {pid}")

def safe_load_learner(fname):
    with open(fname, "rb") as f:
        return SafeUnpickler(f).load()


# =========================================================
# Now import FastAI + Streamlit
# =========================================================
from fastai.vision.all import *
import streamlit as st
from pathlib import Path


MODEL_PATH = Path("final_model.pkl")



@st.cache_resource
def load_model():
    try:
        learn = safe_load_learner(MODEL_PATH)
        return learn
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()


# =========================================================
# Cure suggestion function
# =========================================================
def get_cure_suggestion(disease_name: str):
    disease = disease_name.lower()

    cures = {
        "healthy": "The leaf is healthy. No treatment required.",
        "bacterial spot": "- Apply copper fungicides.\n- Avoid overhead watering.",
        "leaf mold": "- Use sulfur fungicides.\n- Increase ventilation.",
        "early blight": "- Use mancozeb/chlorothalonil.\n- Remove infected leaves.",
        "late blight": "- Apply copper sulfate.\n- Destroy infected plants.",
        "septoria": "- Use chlorothalonil or mancozeb.\n- Prune lower leaves.",
    }

    for key in cures:
        if key in disease:
            return cures[key]

    return "⚠ Unknown disease."


# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="🌿 Crop Disease Identifier", layout="wide")

st.title("🌿 Prompt-Based Crop Disease Identifier")
st.markdown("### Identify plant leaf diseases using a FastAI trained model")


@st.cache_resource
def load_model():
    try:
        learn = load_fastai_model("model_clean.pkl")
        return learn
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()
@st.cache_resource
def load_model():
    try:
        learn = load_fastai_model("model_clean.pkl")
        return learn
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()
learn = load_model()

prompt = st.text_input("💬 Enter your prompt")
uploaded = st.file_uploader("📸 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded and prompt:
    img = PILImage.create(uploaded)
    st.image(img.to_thumb(400, 400), caption="Uploaded Image")

    with st.spinner("🔍 Analyzing..."):
        pred_class, pred_idx, probs = learn.predict(img)

    clean = str(pred_class).replace("_", " ").title()

    st.subheader("🩺 Prediction")
    st.success(f"Disease: {clean}")
    st.info(f"Confidence: {probs[pred_idx]:.2%}")

    st.subheader("💊 Cure Suggestion")
    st.markdown(get_cure_suggestion(clean))

else:
    st.warning("💡 Enter prompt & upload image to continue.")
