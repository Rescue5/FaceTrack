import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Tuple

class OneEuroFilter():
    
    def __init__(self, conf: DictConfig | ListConfig):
        
        default_conf = {
            "landmarks": {},
            "blendshapes": {},
            "defaults": {
                "min_cutoff": 1.0,
                "beta": 0.0,
            },
        }
        
        self.conf = OmegaConf.merge(default_conf, conf)

        self.default_min_cutoff = float(self.conf.defaults.min_cutoff)
        self.default_beta = float(self.conf.defaults.beta)

        self.landmark_overrides: dict[int, tuple[float, float]] = self._parse_overrides(self.conf.landmarks)
        self.blendshape_overrides: dict[int, tuple[float, float]] = self._parse_overrides(self.conf.blendshapes)

    def _parse_overrides(self, overrides) -> dict[int, tuple[float, float]]:
        parsed = {}
        
        if overrides is None:
            return parsed

        items = overrides.items() if hasattr(overrides, "items") else []
        for k, v in items:
            idx = int(k)

            try:
                min_cutoff = float(v[0]) if len(v) >= 1 and v[0] is not None else self.default_min_cutoff
                beta = float(v[1]) if len(v) == 2 and v[1] is not None else self.default_beta
                parsed[idx] = (min_cutoff, beta)
            except Exception as e:
                print(f"Invalid override for index {idx}: {v}, using default values")

        return parsed

    def get_landmark_params(self, idx: int) -> tuple[float, float]:
        return self.landmark_overrides.get(int(idx), (self.default_min_cutoff, self.default_beta))

    def get_blendshape_params(self, idx: int) -> tuple[float, float]:
        return self.blendshape_overrides.get(int(idx), (self.default_min_cutoff, self.default_beta))

    def filter(self, data: np.ndarray | Tuple[np.ndarray, np.ndarray]):
        pass