# uvicorn main:app --host 0.0.0.0 --port 8000


from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://192.168.2.103:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session = ort.InferenceSession("artline_fp16_dynamic.onnx")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img: Image.Image, size: int = 512):
    """Center-pad + resize. Returns (tensor, (x_off, y_off, new_w, new_h))."""
    pad = Image.new("RGB", (size, size), (255, 255, 255))
    ratio = min(size / img.width, size / img.height)
    new_w, new_h = int(img.width * ratio), int(img.height * ratio)
    x_off = (size - new_w) // 2
    y_off = (size - new_h) // 2
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    pad.paste(resized, (x_off, y_off))
    arr = np.array(pad).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)[None, ...], (x_off, y_off, new_w, new_h)


def build_svg(binary: np.ndarray, width: int, height: int) -> str:
    """Trace a binary mask into SVG <path> elements via contour detection."""
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    path_parts = []
    for cnt in contours:
        if len(cnt) < 2:
            continue
        pts = cnt.reshape(-1, 2)
        d = f"M {pts[0,0]} {pts[0,1]}"
        for pt in pts[1:]:
            d += f" L {pt[0]} {pt[1]}"
        d += " Z"
        path_parts.append(d)

    path_d = " ".join(path_parts)
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        + (f'<path d="{path_d}" fill="#000000" fill-rule="evenodd" stroke="none"/>'
           if path_d else "")
        + "</svg>"
    )


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    sensitivity: float = Query(default=1.0, ge=0.1, le=2.0,
                               description="Threshold multiplier: >1 catches more lines, <1 fewer"),
    vectorize: bool = Query(default=True,
                            description="Return SVG paths (true) or RGBA PNG (false)"),
):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    orig_w, orig_h = img.width, img.height

    tensor, (x_off, y_off, new_w, new_h) = preprocess(img)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: tensor})[0]
    # output shape: [1, 3, H, W]

    # de-normalize
    rgb = output[0] * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
    rgb = np.clip(rgb, 0.0, 1.0) * 255.0
    rgb = rgb.astype(np.uint8)  # [3, H, W]

    # crop padding back to content region
    rgb = rgb[:, y_off:y_off + new_h, x_off:x_off + new_w]  # [3, new_h, new_w]

    # luma
    r, g, b = rgb
    luma = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

    # Otsu threshold shifted by sensitivity (no pre-blur — preserves thin lines)
    thr_raw, _ = cv2.threshold(luma, 0, 255, cv2.THRESH_OTSU)
    thr = int(np.clip(thr_raw * sensitivity, 0, 255))

    binary = np.where(luma < thr, 255, 0).astype(np.uint8)

    # scale back to original image dimensions
    if (new_w, new_h) != (orig_w, orig_h):
        binary = cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    if vectorize:
        svg = build_svg(binary, orig_w, orig_h)
        return Response(svg.encode(), media_type="image/svg+xml")

    # fallback: RGBA PNG
    mask = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    mask[binary == 255, 3] = 255
    pil_mask = Image.fromarray(mask, mode="RGBA")
    buf = io.BytesIO()
    pil_mask.save(buf, format="PNG")
    buf.seek(0)
    return Response(buf.read(), media_type="image/png")