# uvicorn main:app --host 0.0.0.0 --port 8000


from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

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


def preprocess(img: Image.Image, size: int = 512) -> np.ndarray:
    # center-pad + resize
    pad = Image.new("RGB", (size, size), (255,255,255))
    ratio = min(size/img.width, size/img.height)
    new_w, new_h = int(img.width*ratio), int(img.height*ratio)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    pad.paste(resized, ((size-new_w)//2, (size-new_h)//2))
    arr = np.array(pad).astype(np.float32)/255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2,0,1)[None,...]


def otsu_threshold(luma: np.ndarray) -> int:
    # simple Otsu's threshold
    hist, _ = np.histogram(luma, bins=256, range=(0,255))
    total = luma.size
    sum_all = np.dot(np.arange(256), hist)
    sumB = 0; wB = 0; varMax = 0; threshold = 0
    for t in range(256):
        wB += hist[t]
        if wB == 0: continue
        wF = total - wB
        if wF == 0: break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_all - sumB) / wF
        var_between = wB * wF * (mB - mF)**2
        if var_between > varMax:
            varMax = var_between
            threshold = t
    return threshold


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = preprocess(img)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: tensor})[0]
    # output shape: [1,3,H,W]
    _,C,H,W = output.shape

    # de-normalize
    rgb = (output[0] * IMAGENET_STD[:,None,None] + IMAGENET_MEAN[:,None,None])
    rgb = np.clip(rgb, 0.0, 1.0) * 255.0
    rgb = rgb.astype(np.uint8)      # shape [3,H,W]

    # compute luma
    r,g,b = rgb
    luma = (0.299*r + 0.587*g + 0.114*b).astype(np.uint8)
    thr = otsu_threshold(luma)

    # build RGBA mask
    mask = np.zeros((H,W,4), dtype=np.uint8)
    # mask pixels: luma<thr => opaque black, else transparent
    mask[luma < thr, 3] = 255

    pil_mask = Image.fromarray(mask, mode="RGBA")
    buf = io.BytesIO()
    pil_mask.save(buf, format="PNG")
    buf.seek(0)
    return Response(buf.read(), media_type="image/png")