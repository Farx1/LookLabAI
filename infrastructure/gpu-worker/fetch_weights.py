"""Download FASHN VTON v1.5 + DWPose weights (same files as upstream scripts/download_weights.py,
without triggering the human-parser download). With a second argument, also download the SAM2.1
small checkpoint (Apache-2.0) used by the layered try-on mode."""

import hashlib
import os
import sys
import urllib.request

from huggingface_hub import hf_hub_download

# sha256 of the file first downloaded from the official URL below (184,416,285 bytes)
SAM2_SHA256 = "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"

d = sys.argv[1] if len(sys.argv) > 1 else "/weights"
os.makedirs(os.path.join(d, "dwpose"), exist_ok=True)
hf_hub_download("fashn-ai/fashn-vton-1.5", "model.safetensors", local_dir=d)
for f in ("yolox_l.onnx", "dw-ll_ucoco_384.onnx"):
    hf_hub_download("fashn-ai/DWPose", f, local_dir=os.path.join(d, "dwpose"))
print("weights ready in", d)
if len(sys.argv) > 2:
    os.makedirs(sys.argv[2], exist_ok=True)
    sam2_path = os.path.join(sys.argv[2], "sam2.1_hiera_small.pt")
    urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt", sam2_path)
    with open(sam2_path, "rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != SAM2_SHA256:
            os.remove(sam2_path)
            sys.exit("SAM2 weights: sha256 mismatch, file removed")
    print("SAM2 weights ready in", sys.argv[2])
