import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Header
from dino_sam2.msg import object_info, LabelList, BboxList
from message_filters import ApproximateTimeSynchronizer, Subscriber

import torch
import PIL
import cv2
import os
import ipdb
import numpy as np
from collections import defaultdict

from sam2.sam2_camera_predictor import SAM2CameraPredictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class ProduceMask:
    def __init__(
        self,
        device,
        dino_checkpoint: str = "IDEA-Research/grounding-dino-base",
        sam2_checkpoint: str = "facebook/sam2-hiera-large",
    ):
        rospy.init_node("produce_mask", anonymous=True)

        # subscriber
        self.obj_info_sub = rospy.Subscriber(
            "/object_info",
            object_info,
            self.callback_update_obj_set,
        )
        self.img_sub = rospy.Subscriber(
            "/camera/image_bgr",
            Image,
            self.callback_produce_mask,
        )

        # publisher for mask
        self.label_pub = rospy.Publisher("/label_list", LabelList, queue_size=10)
        self.bbox_pub = rospy.Publisher("/bbox_list", BboxList, queue_size=10)
        self.mask_pub = rospy.Publisher("/camera/mask", Image, queue_size=10)

        self.bridge = CvBridge()
        self.object_set = set()

        # load model
        self.device = device

        self.grounding_processor = AutoProcessor.from_pretrained(dino_checkpoint)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            dino_checkpoint
        ).to(self.device)

        self.mask_predictor: SAM2CameraPredictor = SAM2ImagePredictor.from_pretrained(
            sam2_checkpoint,
            device=self.device,
        )

        rospy.loginfo("Model loaded")

    def callback_update_obj_set(self, obj_info_msg):
        if obj_info_msg.mode == 0:
            self.object_set.add(obj_info_msg.name)
            rospy.loginfo(f"Added object: {obj_info_msg.name}")
        elif obj_info_msg.mode == 1:
            self.object_set.discard(obj_info_msg.name)
            rospy.loginfo(f"Removed object: {obj_info_msg.name}")
        else:
            rospy.logwarn(f"Unknown mode: {obj_info_msg.mode}")

    def callback_produce_mask(self, img_msg):
        if self.object_set:
            start_time = rospy.Time.now()

            frame = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            image = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            text_prompt = ". ".join(self.object_set) + "."

            inputs = self.grounding_processor(
                images=image,
                text=text_prompt,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.grounding_model(**inputs)

            results = self.grounding_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.5,
                text_threshold=0.5,
                target_sizes=[frame.shape[:2]],
            )

            labels: list = results[0]["labels"]
            input_boxes: torch.Tensor = results[0]["boxes"]

            # sam2
            self.mask_predictor.set_image(image)

            masks, _, _ = self.mask_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

            header = Header()
            header.stamp = rospy.Time.now()
            # publish label list
            label_list = LabelList()
            label_list.data = labels
            label_list.header = header
            self.label_pub.publish(label_list)

            # publish bbox list
            bbox_list = BboxList()
            bbox_list.data = input_boxes.cpu().numpy().flatten().tolist()
            bbox_list.header = header
            self.bbox_pub.publish(bbox_list)

            # publish mask
            masks = masks.astype(np.uint8) * 255
            combined_mask = np.vstack([masks[i, 0] for i in range(len(labels))])
            mask_msg = self.bridge.cv2_to_imgmsg(combined_mask, "mono8")
            mask_msg.header = header
            self.mask_pub.publish(mask_msg)

            rospy.loginfo(f"Frequency: {1 / (rospy.Time.now() - start_time).to_sec()}")

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
        # use bfloat16 for the entire notebook
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        masker = ProduceMask(device=device)
        masker.run()
    except rospy.ROSInterruptException:
        pass
