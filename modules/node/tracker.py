from omegaconf import DictConfig, OmegaConf
from modules.FaceMesh import FaceMeshHandler
import threading
from multiprocessing import Queue
from dataclasses import dataclass
import numpy as np
from modules.node.reader import FramePayload

@dataclass
class TrackerPayload:
    landmarks: np.ndarray
    blanshape: np.ndarray

class Tracker:
    default_conf = OmegaConf.create({
        "sus":True,
    })

    def __init__(self, conf: DictConfig, face_mesh_handler: FaceMeshHandler, frame_queue: Queue[FramePayload], tracker_queue: Queue[TrackerPayload]):
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.face_mesh_handler = face_mesh_handler
        self.stop_event = threading.Event()
        self.frame_queue = frame_queue
        self.tracker_queue = tracker_queue    

    def start(self):
        while not self.stop_event.is_set():
            frame_payload = self.frame_queue.get()
            result = self.face_mesh_handler.process_frame_3d(frame_payload.frame)
            tracker_payload = TrackerPayload(landmarks=result.landmarks[0], blanshape=result.blanshape[0])
            self.tracker_queue.put(tracker_payload)
    
    def stop(self):
        self.stop_event.set()