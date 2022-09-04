import numpy as np
from typing import Tuple, Dict, List
import random

class TimeWarp():
    def __init__(self, keys: List, scale_range: Tuple[float, float] = (0.8, 1.2)) -> None:
        self.keys = keys
        self.scale_range = scale_range
        return

    def __call__(self, data: Dict) -> Dict:
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        for key in self.keys:
            data[key] = data[key]*scale
        return data

class AddRandomNoise():
    def __init__(self, keys: List, mean: float, std: float) -> None:
        self.keys = keys
        self.mean = mean
        self.std = std
        return

    def __call__(self, data: Dict) -> Dict:
        for key in self.keys:
            data[key] = data[key] + np.random.normal(self.mean, self.std, data[key].shape)
        return data

class AddRandomGain():
    def __init__(self, keys: List, gain_range: Tuple[float, float] = (0.8, 1.2)) -> None:
        self.keys = keys
        self.scale_range = gain_range
        return

    def __call__(self, data: Dict) -> Dict:
        gain = random.uniform(self.scale_range[0], self.scale_range[1])
        for key in self.keys:
            data[key] = data[key]*gain
        return data

    def __call__(self, data: Dict) -> Dict:
        for key in self.keys:
            data[key] = data[key]
        return data

class Normalize_magnitude():
    def __init__(self, keys: List) -> None:
        self.keys = keys
        return

    def __call__(self, data: Dict) -> Dict:
        for key in self.keys:
            data[key] = data[key] / data[key].max()
        return data
