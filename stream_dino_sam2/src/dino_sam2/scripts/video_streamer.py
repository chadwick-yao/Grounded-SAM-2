#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import Header
from message_filters import ApproximateTimeSynchronizer, Subscriber
from dino_sam2.msg import LabelList, BboxList
import numpy as np

import supervision as sv
from supervision.draw.color import ColorPalette


class VideoStreamer:
    def __init__(self):
        rospy.init_node("video_streamer", anonymous=True)
        self.image_pub = rospy.Publisher("/camera/image_bgr", Image, queue_size=10)
        # subscriber
        label_sub = Subscriber("/label_list", LabelList)
        bbox_sub = Subscriber("/bbox_list", BboxList)
        mask_sub = Subscriber("/camera/mask", Image)
        self.sub_ts = ApproximateTimeSynchronizer(
            [label_sub, bbox_sub, mask_sub],
            queue_size=10,
            slop=0.05,
        )
        self.sub_ts.registerCallback(self.callback)

        self.bridge = CvBridge()
        self.img_shape = None
        self.labels = None
        self.bboxes = None
        self.masks = None

        self.camera_index = self.find_camera()
        if self.camera_index is None:
            raise RuntimeError("No camera found")
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            rospy.logerr("Cannot open camera")
            raise RuntimeError("Cannot open camera")

        self.rate = rospy.Rate(30)

    def find_camera(self):
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            ret, frame = cap.read()
            if ret:
                self.img_shape = frame.shape
                cap.release()
                return index
            cap.release()
            index += 1
            if index > 10:  # Limit the search to the first 10 indices
                rospy.logerr("No camera found")
                return None

    def callback(
        self,
        label_msg,
        bbox_msg,
        mask_msg,
    ):
        labels = label_msg.data
        num = len(labels)

        bboxes = bbox_msg.data
        bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4)

        combined_mask = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding="mono8")
        height = self.img_shape[0]
        masks = np.array(
            [combined_mask[i * height : (i + 1) * height, :] for i in range(num)]
        )

        self.labels = labels
        self.bboxes = bboxes
        self.masks = masks

    def run(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logerr("Failed to capture image")
                break

            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "camera_frame"

            image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            image_msg.header = header

            self.image_pub.publish(image_msg)

            if (
                self.labels is not None
                and self.bboxes is not None
                and self.masks is not None
            ):
                detections = sv.Detections(
                    xyxy=self.bboxes,  # (n, 4)
                    mask=self.masks.astype(bool),  # (n, h, w)
                    class_id=np.array(list(range(len(self.labels)))),
                )

                box_annotator = sv.BoxAnnotator(color=ColorPalette.DEFAULT)
                annotated_frame = box_annotator.annotate(
                    scene=frame.copy(), detections=detections
                )

                label_annotator = sv.LabelAnnotator(color=ColorPalette.DEFAULT)
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame, detections=detections, labels=self.labels
                )

                mask_annotator = sv.MaskAnnotator(color=ColorPalette.DEFAULT)
                annotated_frame = mask_annotator.annotate(
                    scene=annotated_frame, detections=detections
                )

                cv2.imshow("Camera", annotated_frame)
            else:
                cv2.imshow("Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            self.rate.sleep()

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        streamer = VideoStreamer()
        streamer.run()
    except rospy.ROSInterruptException:
        pass
    except RuntimeError as e:
        rospy.logerr(str(e))
    finally:
        print("Video streamer stopped")
