import os
import urllib.request
import time
import torch
import cv2
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from omegaconf import DictConfig, OmegaConf

class FaceMeshHandler:
    default_conf = OmegaConf.create({
        "model_path": "face_landmarker.task",
        "running_mode": "IMAGE",
        "num_faces":1,
        "output_face_blendshapes":False,
        "output_facial_transformation_matrixes":False,
    })

    def __init__(self, conf: DictConfig):
        self.conf = OmegaConf.merge(self.default_conf, conf)
        MODEL_URL = (
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
            "float16/latest/face_landmarker.task"
        )

        if not os.path.exists(self.conf.model_path):
            urllib.request.urlretrieve(MODEL_URL, self.conf.model_path)

        self.base_options = python.BaseOptions(model_asset_path=self.conf.model_path)
        running_mode = getattr(vision.RunningMode, self.conf.running_mode.upper())
        self.options = vision.FaceLandmarkerOptions(
            base_options=self.base_options,
            running_mode=running_mode,
            num_faces=self.conf.num_faces,
            output_face_blendshapes=self.conf.output_face_blendshapes,
            output_facial_transformation_matrixes=self.conf.output_facial_transformation_matrixes,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(self.options)
    
    def process_frame_3d(self, frame: np.ndarray):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        if self.options.running_mode == vision.RunningMode.VIDEO:
            timestamp_ms = int(time.time() * 1000)
            return self.landmarker.detect_for_video(mp_image, timestamp_ms)
        return self.landmarker.detect(mp_image)