import time
from omegaconf import DictConfig, ListConfig, OmegaConf
from dataclasses import dataclass
import numpy as np
import cv2
import threading
import queue
from .frame_payload import FramePayload
import os

class VideoReader:
    default_conf = OmegaConf.create({
        "video_src":"cv",
        "target_format":"rgb",
        "video_path": "",
        "target_size":"orig",
        "brightness": 1.0
    })

    def __init__(self, conf: DictConfig | ListConfig, frame_queue: queue.Queue):
        self.conf = OmegaConf.merge(self.default_conf, conf)

        self.stop_event = threading.Event()
        self.frame_queue = frame_queue
        
        if self.conf.video_src == "cv":
            self.cap = cv2.VideoCapture(0)
        elif self.conf.video_src == "file":
            if not os.path.exists(self.conf.video_path):
                raise FileNotFoundError(f"Video file not found: {self.conf.video_path}")
            self.cap = cv2.VideoCapture(self.conf.video_path)

    def start(self):
        self.stop_event.clear()

        if self.conf.video_src == "cv":
            self.__start_cv_frame_src()
        elif self.conf.video_src == "file":
            self.__start_cv_frame_src()
    
    def __start_cv_frame_src(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print("Frame not read")
                continue

            if self.conf.target_format == "rgb":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.conf.brightness != 1.0:
                frame = cv2.convertScaleAbs(frame, alpha=self.conf.brightness, beta=0)

            if self.conf.target_size == "orig":
                frame_payload = FramePayload(frame=frame, timestamp=time.monotonic())
            else:
                frame_payload = FramePayload(frame=cv2.resize(frame, self.conf.target_size), timestamp=time.monotonic())
            
            try:
                self.frame_queue.put(frame_payload, block=False)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put(frame_payload, block=False)
   
    def stop(self):
        self.stop_event.set()

        