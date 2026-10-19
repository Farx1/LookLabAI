"""Download FASHN VTON v1.5 + DWPose weights (same files as upstream scripts/download_weights.py,
without triggering the human-parser download)."""

import os
import sys

from huggingface_hub import hf_hub_download

d = sys.argv[1] if len(sys.argv) > 1 else "/weights"
os.makedirs(os.path.join(d, "dwpose"), exist_ok=True)
hf_hub_download("fashn-ai/fashn-vton-1.5", "model.safetensors", local_dir=d)
for f in ("yolox_l.onnx", "dw-ll_ucoco_384.onnx"):
    hf_hub_download("fashn-ai/DWPose", f, local_dir=os.path.join(d, "dwpose"))
print("weights ready in", d)
