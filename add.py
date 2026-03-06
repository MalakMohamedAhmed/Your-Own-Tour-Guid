"""
add.py — Convert model/data/data.pkl → model.pth
Handles raw state_dict, checkpoint dicts, and bare tensors.
Run from the project root: python add.py
"""

import os, sys, pickle, collections
from pathlib import Path

ROOT     = Path(__file__).parent
PKL_PATH = ROOT / "model" / "data.pkl"
OUT_PATH = ROOT / "model.pth"

if not PKL_PATH.exists():
    print(f"[ERROR] Not found: {PKL_PATH}")
    print("  Make sure you run this from D:\\dotby\\ragy\\")
    sys.exit(1)

print(f"[1/4] Loading {PKL_PATH} ...")

# ── Try torch first (handles PyTorch-specific pickles) ────────────────────────
try:
    import torch
    obj = torch.load(str(PKL_PATH), map_location="cpu", weights_only=False)
    print(f"      torch.load OK  — type: {type(obj).__name__}")
except Exception as e_torch:
    print(f"      torch.load failed ({e_torch}), trying raw pickle …")
    try:
        with open(PKL_PATH, "rb") as f:
            obj = pickle.load(f)
        print(f"      pickle.load OK — type: {type(obj).__name__}")
    except Exception as e_pkl:
        print(f"[ERROR] Both loaders failed:\n  torch : {e_torch}\n  pickle: {e_pkl}")
        sys.exit(1)

import torch  # ensure imported after potential first-try failure

# ── Unwrap common checkpoint wrappers ─────────────────────────────────────────
print(f"[2/4] Inspecting object …")

def unwrap_state_dict(obj):
    """Recursively unwrap common checkpoint structures to get a state_dict."""
    if isinstance(obj, dict):
        keys = set(obj.keys())
        # Standard wrappers
        for key in ("model", "state_dict", "model_state_dict", "net", "weights"):
            if key in keys:
                print(f"      Found wrapper key '{key}', unwrapping …")
                return unwrap_state_dict(obj[key])
        # If all values are tensors → it's already a state_dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            print(f"      Looks like a raw state_dict ({len(keys)} keys)")
            return obj
        # OrderedDict with tensor values
        if isinstance(obj, (collections.OrderedDict,)) and all(
            isinstance(v, torch.Tensor) for v in obj.values()
        ):
            return obj
        # Dump top-level keys to help debug
        print(f"      Dict keys: {list(keys)[:15]}")
        # Try to find any tensor-valued sub-dict
        for k, v in obj.items():
            if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                print(f"      Using sub-key '{k}' as state_dict")
                return v
        return obj  # return as-is; torch.save will still write it
    return obj

state_dict = unwrap_state_dict(obj)

# ── Show a few keys so user can verify ───────────────────────────────────────
if isinstance(state_dict, dict):
    sample_keys = list(state_dict.keys())[:6]
    print(f"      Sample keys: {sample_keys}")
else:
    print(f"      Object is not a dict — saving as-is ({type(state_dict).__name__})")

# ── Save ─────────────────────────────────────────────────────────────────────
print(f"[3/4] Saving → {OUT_PATH} …")
torch.save(state_dict, str(OUT_PATH))
sz = OUT_PATH.stat().st_size / (1024 * 1024)
print(f"      Saved  {sz:.1f} MB")

# ── Quick load-back test ──────────────────────────────────────────────────────
print(f"[4/4] Verifying model.pth loads correctly …")
try:
    import torchvision.models as tv
    import torch.nn as nn

    NUM_CLASSES = 20  # must match your training
    model = tv.swin_t()
    model.head = nn.Linear(model.head.in_features, NUM_CLASSES)

    loaded = torch.load(str(OUT_PATH), map_location="cpu", weights_only=False)
    if isinstance(loaded, dict) and "model" in loaded:
        loaded = loaded["model"]
    elif isinstance(loaded, dict) and "state_dict" in loaded:
        loaded = loaded["state_dict"]

    missing, unexpected = model.load_state_dict(loaded, strict=False)
    if missing:
        print(f"  ⚠  Missing  keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  ⚠  Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    if not missing and not unexpected:
        print("  ✅  model.load_state_dict — perfect match!")
    else:
        print("  ⚠  Partial match — the app will still try to load with strict=False")
except Exception as e:
    print(f"  ⚠  Verification skipped: {e}")

print("\n✅  Done!  Place model.pth next to app.py and run:  streamlit run app.py")