"""Measurement counters for one replication.

All statistics are restricted to the observation window [t0, t1):

* time averages (occupancy, busy servers, server state, class composition,
  joint state) are weighted by the time spent in each state, with every
  interval clipped to the window;
* arrival counts, blocking, and the states seen by arrivals use only arrivals
  whose epoch lies in the window;
* customer averages (W, T, service-time sample moments) use only admitted
  customers that arrived in the window, followed until they depart even if that
  happens after t1.

Occupancy bins are 0..15 plus one overflow bin for n >= 16.
"""
import numpy as np

NBINS = 17            # bins 0..15 and ">=16"
OVERFLOW = 16


def bin_index(n):
    return n if n < OVERFLOW else OVERFLOW


class WindowStats:
    __slots__ = (
        "t0", "t1", "H", "n_classes", "track_joint", "last",
        "occ_time", "area_n", "area_nq", "area_busy", "time_serving", "time_vacation",
        "area_class", "joint_time",
        "offered", "blocked", "admitted", "waited",
        "seen_offered", "seen_admitted", "seen_post",
        "pending", "rec_class", "rec_arrival", "rec_start", "rec_depart", "rec_service",
    )

    def __init__(self, t0, t1, n_classes=1, track_joint=False):
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.H = self.t1 - self.t0
        self.n_classes = n_classes
        self.track_joint = track_joint
        self.last = 0.0
        self.occ_time = [0.0] * NBINS
        self.area_n = 0.0
        self.area_nq = 0.0
        self.area_busy = 0.0
        self.time_serving = 0.0
        self.time_vacation = 0.0
        self.area_class = [0.0] * n_classes
        self.joint_time = {}
        self.offered = [0] * n_classes
        self.blocked = [0] * n_classes
        self.admitted = [0] * n_classes
        self.waited = 0
        self.seen_offered = [0] * NBINS
        self.seen_admitted = [0] * NBINS
        self.seen_post = [0] * NBINS
        self.pending = 0
        self.rec_class = []
        self.rec_arrival = []
        self.rec_start = []
        self.rec_depart = []
        self.rec_service = []

    # ------------------------------------------------------------------ time
    def accumulate(self, now, n, busy, on_vacation, n_class):
        """Credit the state that held during [last, now) and move `last` to now.

        Must be called BEFORE every state change at time `now`.
        """
        last = self.last
        if now > last:
            a = last if last > self.t0 else self.t0
            b = now if now < self.t1 else self.t1
            if b > a:
                dt = b - a
                self.occ_time[n if n < OVERFLOW else OVERFLOW] += dt
                self.area_n += n * dt
                self.area_busy += busy * dt
                self.area_nq += (n - busy) * dt
                if busy:
                    self.time_serving += dt
                if on_vacation:
                    self.time_vacation += dt
                if self.n_classes > 1:
                    for i in range(self.n_classes):
                        self.area_class[i] += n_class[i] * dt
                    if self.track_joint:
                        key = tuple(n_class)
                        self.joint_time[key] = self.joint_time.get(key, 0.0) + dt
            self.last = now

    def in_window(self, t):
        return self.t0 <= t < self.t1

    # -------------------------------------------------------------- arrivals
    def record_offered(self, cls, n_seen):
        self.offered[cls] += 1
        self.seen_offered[n_seen if n_seen < OVERFLOW else OVERFLOW] += 1

    def record_blocked(self, cls):
        self.blocked[cls] += 1

    def record_admitted(self, cls, n_seen, n_after, immediate):
        self.admitted[cls] += 1
        self.seen_admitted[n_seen if n_seen < OVERFLOW else OVERFLOW] += 1
        self.seen_post[n_after if n_after < OVERFLOW else OVERFLOW] += 1
        if not immediate:
            self.waited += 1
        self.pending += 1

    def record_departure(self, cust, now):
        self.pending -= 1
        self.rec_class.append(cust.cls)
        self.rec_arrival.append(cust.arrival)
        self.rec_start.append(cust.start)
        self.rec_depart.append(now)
        self.rec_service.append(cust.service)

    # --------------------------------------------------------------- summary
    def summary(self, keep_records=False):
        H = self.H
        arr = np.asarray(self.rec_arrival)
        start = np.asarray(self.rec_start)
        dep = np.asarray(self.rec_depart)
        svc = np.asarray(self.rec_service)
        cls = np.asarray(self.rec_class)
        n_rec = len(arr)
        offered = sum(self.offered)
        blocked = sum(self.blocked)
        admitted = sum(self.admitted)
        p = np.asarray(self.occ_time) / H
        W = float(np.mean(start - arr)) if n_rec else float("nan")
        T = float(np.mean(dep - arr)) if n_rec else float("nan")
        lam_adm = admitted / H
        out = {
            "window_start_s": self.t0,
            "window_length_s": H,
            "offered": offered,
            "blocked": blocked,
            "admitted": admitted,
            "n_delay_samples": n_rec,
            "offered_rate": offered / H,
            "admitted_rate": lam_adm,
            "P_block": blocked / offered if offered else float("nan"),
            "P_Q": self.waited / admitted if admitted else float("nan"),
            "N": self.area_n / H,
            "N_Q": self.area_nq / H,
            "busy_mean": self.area_busy / H,
            "frac_serving": self.time_serving / H,
            "frac_vacation": self.time_vacation / H,
            "frac_empty": p[0],
            "W": W,
            "T": T,
            "little_N_from_T": lam_adm * T,
            "little_NQ_from_W": lam_adm * W,
            "svc_mean": float(np.mean(svc)) if n_rec else float("nan"),
            "svc_second_moment": float(np.mean(svc ** 2)) if n_rec else float("nan"),
            "prob_sum": float(p.sum()),
        }
        for i in range(self.n_classes):
            out[f"offered_c{i+1}"] = self.offered[i]
            out[f"blocked_c{i+1}"] = self.blocked[i]
            out[f"admitted_c{i+1}"] = self.admitted[i]
            out[f"P_block_c{i+1}"] = (self.blocked[i] / self.offered[i]
                                     if self.offered[i] else float("nan"))
            out[f"admitted_rate_c{i+1}"] = self.admitted[i] / H
            if self.n_classes > 1:
                out[f"N_c{i+1}"] = self.area_class[i] / H
                mask = cls == i
                out[f"T_c{i+1}"] = float(np.mean((dep - arr)[mask])) if mask.any() else float("nan")
        hist = {
            "time_avg": p,
            "seen_offered": _normalize(self.seen_offered),
            "seen_admitted": _normalize(self.seen_admitted),
            "seen_post": _normalize(self.seen_post),
        }
        joint = {k: v / H for k, v in self.joint_time.items()} if self.track_joint else None
        res = {"metrics": out, "hist": hist, "joint": joint}
        if keep_records:
            res["records"] = {"class": cls, "arrival": arr, "start": start,
                              "depart": dep, "service": svc}
        return res


def _normalize(counts):
    c = np.asarray(counts, dtype=float)
    s = c.sum()
    return c / s if s > 0 else c
