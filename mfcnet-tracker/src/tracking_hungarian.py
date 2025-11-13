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
    missed: int = 0  # 连续未匹配帧数

class KeypointTracker:
    """
    跟踪某一类关键点（如 tip）。按帧 update，内部用匈牙利做最小距离匹配。
    """
    def __init__(self, max_match_dist=20.0, max_age=8, spawn_score_thresh=0.1):
        self.max_match_dist = float(max_match_dist)   # 距离门限（像素）
        self.max_age = int(max_age)                   # 最多丢多少帧不删轨迹
        self.spawn_score_thresh = float(spawn_score_thresh)  # 新建轨迹分数门限
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
        dist[dist > self.max_match_dist] = _BIG  # 超过门限的配对禁用
        return dist

    def update(self, dets_xy, dets_score):
        """
        dets_xy: (N,2) np.float32
        dets_score: (N,) np.float32
        """
        T = len(self.tracks); M = len(dets_xy)

        # 1) 没历史轨，就把高分检测建成新轨
        if T == 0:
            for i in range(M):
                if dets_score[i] >= self.spawn_score_thresh:
                    self.tracks.append(Track(self._next_id,
                                             dets_xy[i].astype(np.float32).copy(),
                                             float(dets_score[i])))
                    self._next_id += 1
            return

        # 2) 没检测到，历史轨 missed+1
        if M == 0:
            for t in self.tracks:
                t.missed += 1
            self.tracks = [t for t in self.tracks if t.missed <= self.max_age]
            return

        # 3) 计算代价矩阵 + 匈牙利
        C = self._cost_matrix(dets_xy)   # (T,M)
        if _HAVE_SCIPY and C.size > 0:
            rows, cols = linear_sum_assignment(C)
        else:
            # 简单贪心替代（没有 scipy 也能跑）
            rows, cols = [], []
            Cg = C.copy()
            while Cg.size > 0 and np.isfinite(Cg.min()) and Cg.min() < _BIG:
                i, j = np.unravel_index(Cg.argmin(), Cg.shape)
                rows.append(i); cols.append(j)
                Cg[i, :] = _BIG
                Cg[:, j] = _BIG
            rows, cols = np.array(rows, int), np.array(cols, int)

        # 4) 应用匹配结果
        matched_tracks, matched_dets = set(), set()
        for i, j in zip(rows, cols):
            if C[i, j] >= _BIG:
                continue  # 超门限当作未匹配
            t = self.tracks[i]
            t.xy = dets_xy[j].astype(np.float32).copy()
            t.score = float(dets_score[j])
            t.missed = 0
            matched_tracks.add(i)
            matched_dets.add(j)

        # 5) 没匹配上的历史轨 missed+1
        for ti in range(T):
            if ti not in matched_tracks:
                self.tracks[ti].missed += 1

        # 6) 没匹配上的检测、且分数够，生成新轨
        for dj in range(M):
            if dj not in matched_dets and dets_score[dj] >= self.spawn_score_thresh:
                self.tracks.append(Track(self._next_id,
                                         dets_xy[dj].astype(np.float32).copy(),
                                         float(dets_score[dj])))
                self._next_id += 1

        # 7) 清理超期轨迹
        self.tracks = [t for t in self.tracks if t.missed <= self.max_age]

    def active(self):
        return [t for t in self.tracks if t.missed == 0]

class MultiClassTracker:
    """
    管三类：tip / anchor / contact。每帧只需喂一个 dict 进来。
    """
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
        # arr: (N,3) [x, y, score]，兼容空
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
