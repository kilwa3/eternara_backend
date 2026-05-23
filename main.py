# uvicorn main:app --host 0.0.0.0 --port 8000


from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import cv2
from rembg import remove as rembg_remove, new_session as rembg_new_session

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://192.168.2.103:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session = ort.InferenceSession("artline_fp16_dynamic.onnx")
rembg_session = rembg_new_session("isnet-general-use")
REMBG_MAX_SIZE = 1024  # cap rembg input to keep CPU inference fast


ALPHA_GAMMA = 0.35        # aggressive boost for semi-transparent hair
ALPHA_BG_CUTOFF = 20 / 255  # alpha values below this are treated as background


def remove_bg_scaled(img: Image.Image, alpha_gamma: float = ALPHA_GAMMA, **kwargs) -> Image.Image:
    """Run rembg at capped resolution, then scale alpha mask back to original size.
    alpha_gamma < 1 boosts semi-transparent hair pixels toward full opacity."""
    orig_size = img.size
    ratio = min(REMBG_MAX_SIZE / img.width, REMBG_MAX_SIZE / img.height, 1.0)
    small = img.resize(
        (int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS
    ) if ratio < 1.0 else img
    result_small = rembg_remove(small.convert("RGBA"), session=rembg_session, **kwargs)

    # Morphological hair fix:
    # 1. core = pixels ISNet is confident are foreground (alpha > 50%)
    # 2. dilate core to capture nearby hair strands
    # 3. boost alpha inside the dilated zone; zero everything outside
    arr = np.array(result_small, dtype=np.float32)
    a_raw = arr[:, :, 3]
    core = (a_raw > 128).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    dilated = cv2.dilate(core, kernel)
    a_norm = a_raw / 255.0
    boosted = np.power(np.clip(a_norm, 1e-6, 1.0), alpha_gamma)
    arr[:, :, 3] = np.where(dilated > 0, boosted * 255.0, 0.0)
    result_small = Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode="RGBA")

    return result_small.resize(orig_size, Image.LANCZOS) if ratio < 1.0 else result_small


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


def build_svg(binary: np.ndarray, width: int, height: int, min_area: int = 15, smoothing: float = 1.5) -> str:
    """Trace a binary mask into SVG <path> elements via contour detection."""
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    path_parts = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        cnt = cv2.approxPolyDP(cnt, smoothing, True)
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
    remove_background: bool = Query(default=False,
                                    description="Remove background before sketching"),
    min_area: int = Query(default=15,
                          description="Minimum contour area in pixels to include in SVG"),
    smoothing: float = Query(default=1.5,
                             description="approxPolyDP epsilon for contour smoothing"),
):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    if remove_background:
        img_rgba = remove_bg_scaled(img)
        # composite subject onto white canvas
        white = Image.new("RGB", img_rgba.size, (255, 255, 255))
        white.paste(img_rgba, mask=img_rgba.split()[3])
        img = white

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
        svg = build_svg(binary, orig_w, orig_h, min_area=min_area, smoothing=smoothing)
        return Response(svg.encode(), media_type="image/svg+xml")

    # fallback: RGBA PNG
    mask = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    mask[binary == 255, 3] = 255
    pil_mask = Image.fromarray(mask, mode="RGBA")
    buf = io.BytesIO()
    pil_mask.save(buf, format="PNG")
    buf.seek(0)
    return Response(buf.read(), media_type="image/png")


@app.post("/remove-bg")
async def remove_bg(
    file: UploadFile = File(...),
    alpha_matting: bool = Query(default=False,
                                description="Refine edges with alpha matting (slower, better for hair/fur)"),
):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    result = remove_bg_scaled(
        img,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
    )
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return Response(buf.read(), media_type="image/png")