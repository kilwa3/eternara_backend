"""
Informative Drawings inference module.
Architecture from https://github.com/carolineec/informative-drawings
"""

import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image


def load_model(weights_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(weights_path)


def run_inference(img: Image.Image, session: ort.InferenceSession) -> Image.Image:
    """Run model on a PIL RGB image. Returns a white-bg grayscale PIL image (dark = lines)."""
    arr = np.array(img.resize((512, 512), Image.LANCZOS)).astype(np.float32) / 127.5 - 1.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)
    out = session.run(None, {"input": arr})[0][0, 0]  # (H, W)
    out = ((out + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out).resize((img.width, img.height), Image.LANCZOS).convert("RGB")


def to_binary(pil_img: Image.Image,
              dilation_px: int = 0,
              close_px: int = 0) -> np.ndarray:
    """Convert a white-bg sketch image to a binary mask (255 = lines, 0 = background).

    Applies CLAHE contrast enhancement before Otsu threshold, then optionally
    closes gaps and dilates lines.
    """
    gray = np.array(pil_img.convert("L"))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    gray = clahe.apply(gray)

    thr_raw, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    thr = int(thr_raw) if int(thr_raw) > 0 else 128
    binary = np.where(gray < thr, 255, 0).astype(np.uint8)

    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (close_px * 2 + 1, close_px * 2 + 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    if dilation_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (dilation_px * 2 + 1, dilation_px * 2 + 1))
        binary = cv2.dilate(binary, k)

    return binary
