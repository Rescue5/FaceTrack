from omegaconf import DictConfig, ListConfig, OmegaConf
from modules.FaceMesh import FaceMeshHandler
import threading
import queue
from dataclasses import dataclass
import numpy as np
import multiprocessing
from modules.robust.one_euro import OneEuroFilter
from .tracker_payload import TrackerPayload

class Tracker:
    default_conf = OmegaConf.create({
        "store_frame":False,
        "use_one_euro": False,
    })

    def __init__(self, conf: DictConfig | ListConfig, 
                 face_mesh_handler: FaceMeshHandler, 
                 frame_queue: queue.Queue, 
                 tracker_queue: multiprocessing.Queue,
                 one_euro_filter: OneEuroFilter|None):
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.face_mesh_handler = face_mesh_handler
        self.stop_event = threading.Event()
        self.frame_queue = frame_queue
        self.tracker_queue = tracker_queue    
        self.one_euro_filter = one_euro_filter

    def start(self):
        prev_state = "LOST"
        state_num = 0
        while not self.stop_event.is_set():
            frame_payload = self.frame_queue.get()
            tracker_payload = TrackerPayload(None, np.array([]), np.array([]), "", 0, 0.0)

            result = self.face_mesh_handler.process_frame_3d(frame_payload.frame)
            
            prev_state, state_num = self.__return_state(result, prev_state, state_num)

            if prev_state == "LOST":
                tracker_payload.landmarks = np.array([])
                tracker_payload.blanshape = np.array([])
            else:
                landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in result.face_landmarks[0]], dtype=np.float32)
                blendshapes_array = np.array([bs.score for bs in result.face_blendshapes[0]], dtype=np.float32)
                tracker_payload.landmarks = landmarks_array
                tracker_payload.blanshape = blendshapes_array

            if self.conf.store_frame:
                tracker_payload.frame = frame_payload.frame

            tracker_payload.state = prev_state
            tracker_payload.state_num = state_num
            tracker_payload.timestamp = frame_payload.timestamp
            
            if self.conf.use_one_euro and self.one_euro_filter is not None:
                tracker_payload = self.one_euro_filter.filter(tracker_payload)

            self.tracker_queue.put(tracker_payload)
    
    def stop(self):
        self.stop_event.set()

    def __return_state(self, result, prev_state: str, state_num: int) -> tuple[str, int]:
        new_state = prev_state
        has_face = bool(result.face_landmarks) and bool(result.face_blendshapes)

        if not has_face:
            new_state = "LOST"
        else:
            if prev_state == "LOST":
                new_state = "REFOUND"
            else:
                new_state = "TRACKING"
        
        if new_state == prev_state:
            state_num += 1
        else:
            state_num = 0
        
        return new_state, state_num
