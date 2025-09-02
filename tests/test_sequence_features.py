import numpy as np
from src.features.sequence_features import SequenceFeatureExtractor

def make_base_hand():
    # 21 points around origin
    pts = np.zeros((21,3), dtype=np.float32)
    # put fingers at increasing x
    pts[4,0] = 0.1
    pts[8,0] = 0.2
    pts[12,0] = 0.3
    pts[16,0] = 0.4
    pts[20,0] = 0.5
    return pts

def synth_seq(T=15, direction=+1):
    base = make_base_hand()
    seq = np.repeat(base[None,:,:], T, axis=0)
    dx = direction * (0.1 / (T - 1))
    for t in range(T):
        seq[t,:,0] += dx * t
    return seq

def test_left_right_signs():
    T = 15
    fe = SequenceFeatureExtractor(T)
    seq_r = synth_seq(T, +1)
    f_r, names = fe(seq_r)
    seq_l = synth_seq(T, -1)
    f_l, _ = fe(seq_l)
    # net_dx opposite signs
    idx = names.index("net_dx")
    assert f_r[idx] > 0 and f_l[idx] < 0
    # disp_ratio relatively high
    idx2 = names.index("disp_ratio")
    assert f_r[idx2] > 0.6 and f_l[idx2] > 0.6
