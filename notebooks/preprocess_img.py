import pathlib

import cv2
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

odom_dir = pathlib.Path("/home/chadwick/Downloads/odom")
image_dir = pathlib.Path("/home/chadwick/Downloads/image")

image_file_ls = list(image_dir.glob("*.png"))
odom_file_ls = list(odom_dir.glob("*.pkl"))


def get_timestamp_from_filename(filename):
    return int(filename.stem)


def load_odom_data(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


matched_data = []

for image_file in tqdm(image_file_ls):
    image_timestamp = get_timestamp_from_filename(image_file)
    closest_odom_file = min(
        odom_file_ls,
        key=lambda x: abs(get_timestamp_from_filename(x) - image_timestamp),
    )
    odom_data = load_odom_data(closest_odom_file)
    R_body_to_world = Rotation.from_quat(odom_data["orientation"]).as_matrix()

    cv_image = cv2.imread(str(image_file))
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    height, width, _ = rgb_image.shape

    yaw = np.arctan2(R_body_to_world[1, 0], R_body_to_world[0, 0])
    yaw_pixel_offset = int((yaw / (2 * np.pi)) * width)

    corrected_image = np.roll(rgb_image, shift=-yaw_pixel_offset, axis=1)
    corrected_bgr_image = cv2.cvtColor(corrected_image, cv2.COLOR_RGB2BGR)

    save_path = "/home/chadwick/Downloads/image_mod/" + f"{image_file.stem}.png"
    cv2.imwrite(save_path, corrected_bgr_image)
