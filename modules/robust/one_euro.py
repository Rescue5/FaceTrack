import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Tuple
import time
from modules.node.tracker.tracker_payload import TrackerPayload

class OneEuroFilter():
    
    def __init__(self, conf: DictConfig | ListConfig):
        
        default_conf = {
            "landmarks": {},
            "blendshapes": {},
            "defaults": {
                "min_cutoff": 1.0,
                "beta": 0.0,
                "d_cutoff": 1.0,
            },
        }
        
        self.conf = OmegaConf.merge(default_conf, conf)

        self.default_min_cutoff = float(self.conf.defaults.min_cutoff)
        self.default_beta = float(self.conf.defaults.beta)
        self.default_d_cutoff = float(self.conf.defaults.d_cutoff)

        self.landmark_overrides: dict[int, tuple[float, float]] = self._parse_overrides(self.conf.landmarks)
        self.blendshape_overrides: dict[int, tuple[float, float]] = self._parse_overrides(self.conf.blendshapes)
        
        self.prev_lm: np.ndarray | None = None
        self.prev_dlm: np.ndarray | None = None
        
        self.prev_bs: np.ndarray | None = None
        self.prev_dbs: np.ndarray | None = None

        self.prev_t: float | None = None

    def _alpha(self, cutoff_hz: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff_hz)
        return 1.0 / (1.0 + tau / dt)

    def _lowpass(self, x: np.ndarray, x_prev: np.ndarray, a: float) -> np.ndarray:
        return a * x + (1.0 - a) * x_prev

    def _step_one_euro(
        self,
        x: np.ndarray,
        prev_x: np.ndarray,
        prev_dx: np.ndarray,
        dt: float,
        min_cutoff: float,
        beta: float,
        d_cutoff: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        dx = (x - prev_x) / dt
        a_d = self._alpha(d_cutoff, dt)
        dx_hat = self._lowpass(dx, prev_dx, a_d)

        cutoff = min_cutoff + beta * np.abs(dx_hat)
        a = self._alpha(float(np.max(cutoff)), dt)
        x_hat = self._lowpass(x, prev_x, a)
        return x_hat, dx_hat

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

    def filter(self, tracker_payload: TrackerPayload):
        landmarks, blendshapes = tracker_payload.landmarks, tracker_payload.blanshape
        timestamp = tracker_payload.timestamp

        # Skip filtering if no data
        if len(landmarks) == 0 or len(blendshapes) == 0:
            return tracker_payload

        t = time.monotonic() if timestamp is None else float(timestamp)

        # landmarks and blendshapes are already numpy arrays
        x_lm = np.asarray(landmarks, dtype=np.float32)
        x_bs = np.asarray(blendshapes, dtype=np.float32)

        if self.prev_t is None or self.prev_lm is None or self.prev_dlm is None or self.prev_bs is None or self.prev_dbs is None:
            self.prev_t = t
            self.prev_lm = x_lm
            self.prev_dlm = np.zeros_like(x_lm)
            self.prev_bs = x_bs
            self.prev_dbs = np.zeros_like(x_bs)
            return tracker_payload

        dt = max(1e-6, t - self.prev_t)

        lm_hat, dlm_hat = self._step_one_euro(
            x_lm,
            self.prev_lm,
            self.prev_dlm,
            dt,
            self.default_min_cutoff,
            self.default_beta,
            self.default_d_cutoff,
        )
        
        bs_hat, dbs_hat = self._step_one_euro(
            x_bs,
            self.prev_bs,
            self.prev_dbs,
            dt,
            self.default_min_cutoff,
            self.default_beta,
            self.default_d_cutoff,
        )

        for idx, (min_cutoff, beta) in self.landmark_overrides.items():
            if 0 <= idx < x_lm.shape[0]:
                lm_hat[idx], dlm_hat[idx] = self._step_one_euro(
                    x_lm[idx],
                    self.prev_lm[idx],
                    self.prev_dlm[idx],
                    dt,
                    min_cutoff,
                    beta,
                    self.default_d_cutoff,
                )

        for idx, (min_cutoff, beta) in self.blendshape_overrides.items():
            if 0 <= idx < x_bs.shape[0]:
                bs_hat[idx], dbs_hat[idx] = self._step_one_euro(
                    x_bs[idx],
                    self.prev_bs[idx],
                    self.prev_dbs[idx],
                    dt,
                    min_cutoff,
                    beta,
                    self.default_d_cutoff,
                )

        self.prev_t = t
        self.prev_lm = lm_hat
        self.prev_dlm = dlm_hat
        self.prev_bs = bs_hat
        self.prev_dbs = dbs_hat
        
        # Update payload with filtered numpy arrays
        tracker_payload.landmarks = lm_hat
        tracker_payload.blanshape = bs_hat

        return tracker_payload