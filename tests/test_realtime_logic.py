import numpy as np
from src.realtime.smoothing import PredictionSmoother

def test_prediction_smoother_stability():
    sm = PredictionSmoother(window_size=5, confidence_threshold=0.7)
    for _ in range(5):
        sm.add(1, 0.9)
    label, conf = sm.stable()
    assert label == 1
    assert conf >= 0.7
