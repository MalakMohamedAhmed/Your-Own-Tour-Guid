"""
╔══════════════════════════════════════════════════════════════════════════╗
║  PIPELINE 7 — Image Classification (Swin Transformer)                  ║
║                                                                          ║
║  What this file does:                                                    ║
║  ─────────────────────────────────────────────────────────────────────  ║
║  Classifies an uploaded image into one of 20 Egyptian landmark          ║
║  categories using a fine-tuned Swin-T (Swin Transformer Tiny) model.   ║
║                                                                          ║
║  What is Swin-T?                                                         ║
║    Swin Transformer Tiny is a vision backbone from Microsoft Research.  ║
║    Unlike ViT which patches the whole image at once, Swin uses shifted  ║
║    windows of attention — efficient and accurate at 224×224 input.      ║
║    The final linear classification head (model.head) was replaced with  ║
║    a new nn.Linear layer sized to 20 classes and fine-tuned on a        ║
║    custom dataset of Egyptian heritage photos.                          ║
║                                                                          ║
║  Model loading:                                                          ║
║    The weights file model.pth is expected next to this script.          ║
║    It can be a raw state_dict or wrapped in a dict under any of the     ║
║    common keys: "model", "state_dict", "model_state_dict", "net",       ║
║    "weights". The loader tries all of them automatically.               ║
║    strict=False allows the new head to load even if pretrained keys     ║
║    don't match exactly (missing / unexpected keys are logged).          ║
║                                                                          ║
║  Image preprocessing:                                                    ║
║    Standard ImageNet pipeline used during training:                     ║
║      Resize 256 → CenterCrop 224 → ToTensor → Normalise                ║
║    Same mean/std as the original Swin-T training regime.                ║
║                                                                          ║
║  Output (classify_image):                                               ║
║    Returns {"predictions": [{label, confidence}, …]} with top-3 picks. ║
║    Confidence is expressed as a percentage (0–100).                     ║
║    On any failure, returns {"error": "…"} instead.                      ║
║                                                                          ║
║  20 supported landmark classes:                                          ║
║    Abu Simbel, Alexandria Library, Bent Pyramid, Citadel of Qaitbay,   ║
║    Colossi of Memnon, Egyptian Museum, Great Pyramid of Giza,           ║
║    Great Sphinx, Karnak Temple, Khan el-Khalili, Luxor Temple,          ║
║    Medinet Habu, Mortuary Temple of Hatshepsut, Philae Temple,          ║
║    Pyramid of Khafre, Pyramid of Menkaure, Red Pyramid,                ║
║    Step Pyramid of Djoser, Temple of Edfu, Valley of the Kings.        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import io
import os

import streamlit as st

# ── Optional torch / torchvision ─────────────────────────────────────────────
CLASSIFIER_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torchvision.models import swin_t
    from PIL import Image as PILImage
    CLASSIFIER_AVAILABLE = True
except ImportError:
    pass

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_FOLDER     = os.path.join(BASE_DIR, "data")
CLASSIFIER_WEIGHTS = os.path.join(BASE_DIR, "model", "model.pth")

CLASSIFIER_CLASSES = [
    "Abu Simbel", "Alexandria Library", "Bent Pyramid", "Citadel of Qaitbay",
    "Colossi of Memnon", "Egyptian Museum", "Great Pyramid of Giza",
    "Great Sphinx", "Karnak Temple", "Khan el-Khalili", "Luxor Temple",
    "Medinet Habu", "Mortuary Temple of Hatshepsut", "Philae Temple",
    "Pyramid of Khafre", "Pyramid of Menkaure", "Red Pyramid",
    "Step Pyramid of Djoser", "Temple of Edfu", "Valley of the Kings",
]


# ── Model loader ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="𓂀 Loading the Eye of Horus classifier…")
def load_classifier():
    """
    Load the fine-tuned Swin-T model from model.pth.

    The function:
        1. Instantiates the stock torchvision swin_t() architecture.
        2. Replaces the final linear head with a 20-class layer.
        3. Loads the checkpoint from model.pth.
        4. Unwraps common checkpoint wrapper keys if present.
        5. Calls load_state_dict with strict=False so minor mismatches
           (e.g. a version trained with a slightly different torchvision)
           don't crash the app — only missing keys are warned about.
        6. Sets model to eval() mode (disables dropout / batch-norm updates).

    Returns the model on CPU, or None if weights are missing / torch is absent.
    """
    if not CLASSIFIER_AVAILABLE:
        return None
    if not os.path.exists(CLASSIFIER_WEIGHTS):
        return None
    try:
        model     = swin_t()
        model.head = nn.Linear(model.head.in_features, len(CLASSIFIER_CLASSES))

        state = torch.load(CLASSIFIER_WEIGHTS, map_location="cpu", weights_only=False)

        # Unwrap common checkpoint wrapper keys
        for key in ("model", "state_dict", "model_state_dict", "net", "weights"):
            if isinstance(state, dict) and key in state:
                state = state[key]
                break

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            st.warning(f"𓂀 Classifier: {len(missing)} missing weight keys (non-fatal)")

        model.eval()
        return model

    except Exception as exc:
        st.warning(f"𓂀 Could not load classifier: {exc}")
        return None


# ── Image transform (cached at module level) ──────────────────────────────────
_TRANSFORM = None

def _get_transform():
    """
    Return the standard ImageNet preprocessing pipeline.
    Built once and reused — creating Transform objects is cheap but
    keeping a single instance avoids repeated object allocation.

    Steps:
        Resize(256)           — scale shortest edge to 256 px
        CenterCrop(224)       — extract 224×224 centre square
        ToTensor()            — uint8 [0,255] → float32 [0,1], HWC → CHW
        Normalize(mean, std)  — standard ImageNet channel statistics
    """
    global _TRANSFORM
    if _TRANSFORM is None and CLASSIFIER_AVAILABLE:
        _TRANSFORM = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ])
    return _TRANSFORM


# ── Public entry point ────────────────────────────────────────────────────────

def classify_image(image_bytes: bytes) -> dict:
    """
    Run the Swin-T classifier on raw image bytes.

    Parameters
    ----------
    image_bytes : bytes
        Raw image data in any format PIL can open (JPEG, PNG, WebP, etc.).

    Returns
    -------
    dict
        Success: {"predictions": [{"label": str, "confidence": float}, …]}
                 The list contains up to 3 entries sorted by confidence desc.
                 Confidence is a percentage (0.0–100.0).
        Failure: {"error": "reason string"}
    """
    if not CLASSIFIER_AVAILABLE:
        return {"error": "torch / torchvision not installed — run: pip install torch torchvision"}

    model = load_classifier()
    if model is None:
        return {"error": "Classifier weights not found — place model.pth next to the app script"}

    try:
        img       = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        transform = _get_transform()
        tensor    = transform(img).unsqueeze(0)   # add batch dimension: [1, 3, 224, 224]

        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]

        top3 = torch.topk(probs, k=min(3, len(CLASSIFIER_CLASSES)))
        predictions = [
            {
                "label":      CLASSIFIER_CLASSES[idx.item()],
                "confidence": round(prob.item() * 100, 1),
            }
            for prob, idx in zip(top3.values, top3.indices)
        ]
        return {"predictions": predictions}

    except Exception as exc:
        return {"error": str(exc)}
