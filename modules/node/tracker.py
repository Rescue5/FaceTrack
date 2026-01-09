from omegaconf import DictConfig, OmegaConf
from modules.FaceMesh import FaceMeshHandler
import threading
import queue
from dataclasses import dataclass
import numpy as np

@dataclass
class TrackerPayload:
    frame: np.ndarray | None
    landmarks: np.ndarray
    blanshape: np.ndarray
    state: str
    state_num: int

class Tracker:
    default_conf = OmegaConf.create({
        "store_frame":False,
    })

    def __init__(self, conf: DictConfig, face_mesh_handler: FaceMeshHandler, frame_queue: queue.Queue, tracker_queue: queue.Queue):
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.face_mesh_handler = face_mesh_handler
        self.stop_event = threading.Event()
        self.frame_queue = frame_queue
        self.tracker_queue = tracker_queue    

    def start(self):
        prev_state = "LOST"
        state_num = 0
        while not self.stop_event.is_set():
            frame_payload = self.frame_queue.get()
            tracker_payload = TrackerPayload(None, np.array([]), np.array([]), "", 0)

            result = self.face_mesh_handler.process_frame_3d(frame_payload.frame)
            
            prev_state, state_num = self.__return_state(result, prev_state, state_num)

            if prev_state == "LOST":
                pass
            else:
                tracker_payload.landmarks, tracker_payload.blanshape =  result.face_landmarks[0], result.face_blendshapes[0]

            if self.conf.store_frame:
                tracker_payload.frame = frame_payload.frame

            tracker_payload.state = prev_state
            tracker_payload.state_num = state_num

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
