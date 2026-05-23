"""Convert Informative Drawings PTH checkpoints to ONNX (float32, lossless)."""
import torch
import sys

sys.path.insert(0, ".")
from informative_drawings import load_model

STYLES = ["anime_style", "contour_style"]

for style in STYLES:
    pth_path  = f"checkpoints/{style}/netG_A_latest.pth"
    onnx_path = f"checkpoints/{style}.onnx"

    print(f"Exporting {style} ...")
    model = load_model(pth_path)

    dummy = torch.randn(1, 3, 512, 512)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        dynamo=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"},
        },
        opset_version=17,
    )
    print(f"  -> {onnx_path}")

print("All done.")
