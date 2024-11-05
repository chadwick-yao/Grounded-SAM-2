import os

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import supervision as sv
from supervision.draw.color import ColorPalette

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"


def load_models(
    dino_id="IDEA-Research/grounding-dino-base", sam2_id="facebook/sam2-hiera-large"
):
    mask_predictor = SAM2ImagePredictor.from_pretrained(sam2_id, device=device)
    grounding_processor = AutoProcessor.from_pretrained(dino_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(
        device
    )

    return mask_predictor, grounding_processor, grounding_model


mask_predictor, grounding_processor, grounding_model = load_models()


object_list = ["table", "chair", "trash can"]
text_prompt = ". ".join(object_list) + "."
text_prompt


from tqdm import tqdm
import pathlib

img_dir = "/home/chadwick/Downloads/image"
img_dir = pathlib.Path(img_dir)
img_paths = list(img_dir.glob("*.png"))

for img_path in tqdm(img_paths, desc="Processing images"):
    image = Image.open(img_path)
    image = np.array(image.convert("RGB"))
    inputs = grounding_processor(
        images=image,
        text=text_prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = grounding_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.4,
        target_sizes=[image.shape[:2]],
    )

    class_names: list = results[0]["labels"]
    input_boxes = results[0]["boxes"].cpu().numpy()
    confidences = results[0]["scores"].cpu().numpy().tolist()

    mask_predictor.set_image(image)

    masks, _, _ = mask_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    class_ids = np.array(list(range(len(class_names))))

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(class_names, confidences)
    ]

    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids,
    )

    box_annotator = sv.BoxAnnotator(color=ColorPalette.DEFAULT)
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator(color=ColorPalette.DEFAULT)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )

    mask_annotator = sv.MaskAnnotator(color=ColorPalette.DEFAULT)
    annotated_frame = mask_annotator.annotate(
        scene=annotated_frame, detections=detections
    )

    # save annotated frame, numpy array h x w x 3
    try:
        Image.fromarray(annotated_frame).save(
            pathlib.Path("/home/chadwick/Downloads/image_s") / f"{img_path.stem}.png"
        )
    except Exception as e:
        print(f"Error saving image: {e}")

    # save class_names, input_boxes, masks, and confidences into npz file
    np.savez_compressed(
        pathlib.Path("/home/chadwick/Downloads/image_npz") / f"{img_path.stem}.npz",
        labels=class_names,
        bboxes=input_boxes,
        masks=masks,
        confidences=confidences,
    )
