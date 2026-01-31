from dataclasses import dataclass
import numpy as np

@dataclass
class TrackerPayload:
    frame: np.ndarray | None
    landmarks: np.ndarray
    blanshape: np.ndarray
    state: str
    state_num: int
    timestamp: float
