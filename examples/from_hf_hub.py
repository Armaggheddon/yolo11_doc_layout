from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2

DOWNLOAD_PATH = Path(__file__).parent / "models"
SAMPLES_ROOT = Path(__file__).parent.parent / "plots/samples"

available_models = [
    "yolo11n_doc_layout.pt",
    "yolo11s_doc_layout.pt",
    "yolo11m_doc_layout.pt",
]

model_path = hf_hub_download(
    repo_id="Armaggheddon/yolo11-document-layout",
    filename=available_models[0],  # Change index for different models
    repo_type="model",
    local_dir=DOWNLOAD_PATH,
)

# Initialize the model from the downloaded path
model = YOLO(model_path)

images = list(SAMPLES_ROOT.glob("*.png"))
images.sort()

results = [
    model(str(i), conf=0.5, verbose=False, save=False)
    for i in images
]

for i, r in enumerate(results):
    annotated_frame = r[0].plot()
    output_path = SAMPLES_ROOT / f"annotated_{i+1}.png"
    cv2.imwrite(str(output_path), annotated_frame)
    print(f"Saved annotated image to {output_path}")