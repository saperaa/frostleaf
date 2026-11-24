"""Inference on raw input: an image file in, class + confidence out.

Owner: Evaluation & Deployment Engineer.

Must reuse the SAME preprocessing functions as training (import them from
src.data; never copy them), load the saved model from models/, and report
inference time per image and model size on disk.

Usage (once implemented):
    python -m src.predict path/to/leaf.jpg
"""

from __future__ import annotations


def predict(image_path: str) -> tuple[str, float]:
    """Return (predicted class name, confidence) for one image file."""
    raise NotImplementedError("Evaluation Engineer: implement in Week 13")


if __name__ == "__main__":
    import sys

    label, conf = predict(sys.argv[1])
    print(f"{label} ({conf:.1%})")
