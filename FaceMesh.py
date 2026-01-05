import os
import urllib.request
import torch
import cv2
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceMeshHandler:
    def __init__(self):
        self.MODEL_PATH = "face_landmarker.task"
        MODEL_URL = (
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
            "float16/latest/face_landmarker.task"
        )

        if not os.path.exists(self.MODEL_PATH):
            urllib.request.urlretrieve(MODEL_URL, self.MODEL_PATH)

        self.base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        self.options = vision.FaceLandmarkerOptions(
            base_options=self.base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(self.options)
    
    def process_frame_3d(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        return self.landmarker.detect(mp_image)