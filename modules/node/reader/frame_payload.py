

from dataclasses import dataclass
import numpy as np

@dataclass
class FramePayload:
    frame: np.ndarray
    timestamp: float