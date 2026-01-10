from omegaconf import DictConfig, ListConfig, OmegaConf
from dataclasses import dataclass
from multiprocessing import Queue
from modules.node.tracker import TrackerPayload
import threading

class Render:
    default_conf = OmegaConf.create({
        "sus":True,
    })
    def __init__(self, conf: DictConfig | ListConfig, tracker_queue: Queue):
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.tracker_queue = tracker_queue
        self.stop_event = threading.Event()
    
    def start(self):
        self.stop_event.clear()
        while not self.stop_event.is_set():
            tracker_payload = self.tracker_queue.get()
            pass

    def stop(self):
        self.stop_event.set()