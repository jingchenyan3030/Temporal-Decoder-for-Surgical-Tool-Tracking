# src/tracking_hungarian.py
import numpy as np
from dataclasses import dataclass
try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

_BIG = 1e6

@dataclass
class Track:
    track_id: int
    xy: np.ndarray   # (2,)
    score: float
    missed: int = 0  

class KeypointTracker:

    def __init__(self, max_match_dist=20.0, max_age=8, spawn_score_thresh=0.1):
        self.max_match_dist = float(max_match_dist)   
        self.max_age = int(max_age)                  
        self.spawn_score_thresh = float(spawn_score_thresh)  
        self.tracks = []
        self._next_id = 1

    def reset(self):
        self.tracks = []
        self._next_id = 1

    def _cost_matrix(self, dets_xy):
        T = len(self.tracks); M = len(dets_xy)
        if T == 0 or M == 0:
            return np.zeros((T, M), dtype=np.float32)
        track_xy = np.stack([t.xy for t in self.tracks], 0)      # (T,2)
        dist = np.linalg.norm(track_xy[:, None, :] - dets_xy[None, :, :], axis=2)  # (T,M)
        dist[dist > self.max_match_dist] = _BIG  
        return dist

    def update(self, dets_xy, dets_score):
        """
        dets_xy: (N,2) np.float32
        dets_score: (N,) np.float32
        """
        T = len(self.tracks); M = len(dets_xy)

        if T == 0:
            for i in range(M):
                if dets_score[i] >= self.spawn_score_thresh:
                    self.tracks.append(Track(self._next_id,
                                             dets_xy[i].astype(np.float32).copy(),
                                             float(dets_score[i])))
                    self._next_id += 1
            return

    
        if M == 0:
            for t in self.tracks:
                t.missed += 1
            self.tracks = [t for t in self.tracks if t.missed <= self.max_age]
            return

      
        C = self._cost_matrix(dets_xy)   # (T,M)
        if _HAVE_SCIPY and C.size > 0:
            rows, cols = linear_sum_assignment(C)
        else:
          
            rows, cols = [], []
            Cg = C.copy()
            while Cg.size > 0 and np.isfinite(Cg.min()) and Cg.min() < _BIG:
                i, j = np.unravel_index(Cg.argmin(), Cg.shape)
                rows.append(i); cols.append(j)
                Cg[i, :] = _BIG
                Cg[:, j] = _BIG
            rows, cols = np.array(rows, int), np.array(cols, int)

   
        matched_tracks, matched_dets = set(), set()
        for i, j in zip(rows, cols):
            if C[i, j] >= _BIG:
                continue  
            t = self.tracks[i]
            t.xy = dets_xy[j].astype(np.float32).copy()
            t.score = float(dets_score[j])
            t.missed = 0
            matched_tracks.add(i)
            matched_dets.add(j)

     
        for ti in range(T):
            if ti not in matched_tracks:
                self.tracks[ti].missed += 1

     
        for dj in range(M):
            if dj not in matched_dets and dets_score[dj] >= self.spawn_score_thresh:
                self.tracks.append(Track(self._next_id,
                                         dets_xy[dj].astype(np.float32).copy(),
                                         float(dets_score[dj])))
                self._next_id += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.max_age]

    def active(self):
        return [t for t in self.tracks if t.missed == 0]

class MultiClassTracker:

    def __init__(self,
                 tip_cfg   = dict(max_match_dist=20, max_age=8, spawn_score_thresh=0.10),
                 anchor_cfg= dict(max_match_dist=20, max_age=8, spawn_score_thresh=0.10),
                 contact_cfg=dict(max_match_dist=25, max_age=8, spawn_score_thresh=0.50)):
        self.tip     = KeypointTracker(**tip_cfg)
        self.anchor  = KeypointTracker(**anchor_cfg)
        self.contact = KeypointTracker(**contact_cfg)

    def reset(self):
        self.tip.reset(); self.anchor.reset(); self.contact.reset()

    @staticmethod
    def _xy_score(arr):
        if arr is None or arr.size == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)
        return arr[:, :2].astype(np.float32), arr[:, 2].astype(np.float32)

    def update(self, frame_results):
        """
        frame_results: dict
            {
              'tip':     np.ndarray(N1,3)  # [x,y,score]
              'anchor':  np.ndarray(N2,3)
              'contact': np.ndarray(N3,3)
            }
        """
        tips_xy, tips_sc         = self._xy_score(frame_results.get('tip'))
        anchors_xy, anchors_sc   = self._xy_score(frame_results.get('anchor'))
        contacts_xy, contacts_sc = self._xy_score(frame_results.get('contact'))

        self.tip.update(tips_xy, tips_sc)
        self.anchor.update(anchors_xy, anchors_sc)
        self.contact.update(contacts_xy, contacts_sc)

    def get_active(self):
        return {
            'tip':     self.tip.active(),
            'anchor':  self.anchor.active(),
            'contact': self.contact.active(),
        }
