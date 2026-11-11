"""H1 robot library: Clearpath RidgebackFranka (6.1 asset root) driven through the PhysX tensor API.
- base: the asset's virtual holonomic joints (dummy_base_prismatic_x / _y, revolute_z) in the world frame of the robot prim
- arm: Franka DH forward kinematics + damped-least-squares IK in numpy (own IK, checked against PhysX link poses)
- action API (v18 D7/D9/D12/D14): drive_to(x, y, heading, speed, accel), move_to(x, y, z, R or yaw, gripper), wait(s)
Everything here is sim-only: no model calls."""
import math, os
import numpy as np

URL_REL = "/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd"
ARM_MOUNT = np.array([0.30, 0.0, 0.28])   # panda_link0 in base_link frame (probe: h1_probe_rf.log)
# Franka modified-DH (Craig): (a_{i-1}, d_i, alpha_{i-1})
DH = [(0, 0.333, 0), (0, 0, -math.pi / 2), (0, 0.316, math.pi / 2), (0.0825, 0, math.pi / 2),
      (-0.0825, 0.384, -math.pi / 2), (0, 0, math.pi / 2), (0.088, 0, math.pi / 2)]
FLANGE_D = 0.107
SMOOTH_LINE = os.environ.get("H1_SMOOTH", "1") == "1"     # 2026-10-25: straight-line moves were a 1 cm staircase of joint targets (jerk flipped the pill cup)
TCP_D = 0.1034                       # flange -> fingertip centre (the asset's 'endeffector' link)
QLO = np.radians([-168, -103, -168, -170, -168, -3, -168])     # asset limits (h1_probe_rf.log) minus 2 deg
QHI = np.radians([168, 103, 168, 3, 168, 170, 168])
Q_HOME = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.785])
SEEDS = [np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.785]), np.array([0, 0.2, 0, -1.5, 0, 1.8, 0.785]), np.array([0, 0.5, 0, -1.0, 0, 1.5, 0.785]),
         np.array([0, -0.2, 0, -2.4, 0, 2.6, 0.785]), np.array([0, 0.3, 0, -1.2, 0, 3.0, 0.785]), np.array([0, 0.0, 0, -2.0, 0, 2.9, 0.785])]
Q_CARRY = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.785])


def _mdh(a, d, al, th):
    ca, sa, ct, st = math.cos(al), math.sin(al), math.cos(th), math.sin(th)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]])


def fk(q, tcp=True):
    """pose of the TCP (fingertip centre) in the panda_link0 frame, 4x4"""
    T = np.eye(4)
    for (a, d, al), th in zip(DH, q): T = T @ _mdh(a, d, al, th)
    T = T @ _mdh(0, FLANGE_D, 0, 0)
    Rz = _mdh(0, 0, 0, -math.pi / 4)          # hand frame is the flange turned -45 deg
    T = T @ Rz
    if tcp: T = T @ _mdh(0, TCP_D, 0, 0)
    return T


def rot_err(Rc, Rt):
    E = Rt @ Rc.T
    ang = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]]) * 0.5
    return ang


def ik(p_t, R_t, q0, iters=300, tol=1e-4, w_rot=0.5):
    """DLS IK in the link0 frame. Returns (q, pos_err_m, rot_err_rad, ok)"""
    q = np.array(q0, float)
    for _ in range(iters):
        T = fk(q); e = np.r_[p_t - T[:3, 3], w_rot * rot_err(T[:3, :3], R_t)]
        if np.linalg.norm(e[:3]) < tol and np.linalg.norm(e[3:]) < tol * 3: break
        J = np.zeros((6, 7)); h = 1e-6
        for j in range(7):
            dq = q.copy(); dq[j] += h; Tj = fk(dq)
            J[:3, j] = (Tj[:3, 3] - T[:3, 3]) / h; J[3:, j] = w_rot * rot_err(T[:3, :3], Tj[:3, :3]) / h
        lam = 0.02
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
        dq += 0.02 * (Q_HOME - q) - 0.02 * (J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), J @ (Q_HOME - q)))   # null-space pull to home
        q = np.clip(q + np.clip(dq, -0.2, 0.2), QLO, QHI)
    T = fk(q); pe = float(np.linalg.norm(p_t - T[:3, 3])); re = float(np.linalg.norm(rot_err(T[:3, :3], R_t)))
    return q, pe, re, (pe < 2e-3 and re < 0.02)


def R_down(yaw):
    """gripper pointing straight down (TCP z = -world z), fingers' closing axis set by yaw (rad, about world z)"""
    c, s = math.cos(yaw), math.sin(yaw)
    # columns: x (hand x), y (finger closing axis), z (approach)
    return np.array([[c, s, 0], [s, -c, 0], [0, 0, -1]], float)


def R_side(yaw, roll=0.0):
    """gripper pointing horizontally along heading yaw (rad), finger closing axis vertical (roll=0) -> pinches a horizontal bar
    from above and below"""
    R0 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
    c, s = math.cos(roll), math.sin(roll); Rr = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Rz(yaw) @ R0 @ Rr


def R_pitch(yaw, pitch_deg=45.0):
    """gripper approach axis along heading yaw, pitched pitch_deg below horizontal; fingers close in the vertical plane
    that holds the approach axis (pinches a horizontal round bar that runs across the approach direction)"""
    p = math.radians(pitch_deg); c, s = math.cos(p), math.sin(p)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return Rz(yaw) @ Ry @ R_side(0.0)


def Rz(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


class Robot:
    """wraps the articulation view. Call attach() after the timeline plays and physics stepped once."""
    BASE = ["dummy_base_prismatic_x_joint", "dummy_base_prismatic_y_joint", "dummy_base_revolute_z_joint"]
    ARM = [f"panda_joint{i}" for i in range(1, 8)]
    FING = ["panda_finger_joint1", "panda_finger_joint2"]

    def __init__(self, path, step_fn, dt=1 / 60, odo_scale=0.0, odo_head_deg_per_m=0.0):
        """odo_scale: odometry distance error (fraction, e.g. +0.025 reads 2.5% long); odo_head_deg_per_m: heading drift per metre
        driven (v18 D1/D13; the datasheet states the maxima 3% and 0.5 deg/m). 0, 0 = exact odometry."""
        self.path = path; self.step_fn = step_fn; self.dt = dt; self.sim_t = 0.0
        self.contacts_cb = None; self.stop_check = None; self.on_step = None
        self.odo_s = odo_scale; self.odo_h = math.radians(odo_head_deg_per_m); self.odo = None; self._prev_true = None
        self.odo_path_m = 0.0
        self.grip_speed = float(os.environ.get("H1_GRIP_SPEED", "0")) or None     # None = old behaviour (fingers jump to the target)

    def attach(self):
        import omni.physics.tensors as tensors
        from isaacsim.core.simulation_manager import SimulationManager
        SimulationManager.set_backend("numpy")
        self.sv = SimulationManager.get_physics_sim_view()
        if self.sv is None:
            SimulationManager.initialize_physics(); self.sv = SimulationManager.get_physics_sim_view()
        self.sv.set_subspace_roots("/")
        self.av = self.sv.create_articulation_view(self.path)
        names = list(self.av.shared_metatype.dof_names)
        self.names = names
        self.ib = [names.index(n) for n in self.BASE]; self.ia = [names.index(n) for n in self.ARM]; self.ifg = [names.index(n) for n in self.FING]
        self.links = list(self.av.shared_metatype.link_names)
        self.tgt = self.av.get_dof_positions().copy().reshape(-1)
        # the asset's virtual base joints carry joint friction 0.1; PhysX scales joint friction with the joint's constraint wrench, which
        # grows with the distance from the robot prim origin (the whole robot hangs off the first prismatic joint): at x=-40 m the base
        # could not move in x at all. Friction 0 on the three virtual joints.
        fr = self.av.get_dof_friction_coefficients().reshape(1, -1).copy()
        for i in self.ib: fr[0, i] = 0.0
        self.av.set_dof_friction_coefficients(fr.astype(np.float32), np.array([0], dtype=np.int32))
        return names

    def teleport(self, x=None, y=None, th=None, qa=None, fw=None):
        """set joint state AND targets directly (scene set-up only, never an agent action)"""
        q = self.q()
        if x is not None: q[self.ib[0]] = x
        if y is not None: q[self.ib[1]] = y
        if th is not None: q[self.ib[2]] = th
        if qa is not None: q[self.ia] = qa
        if fw is not None: q[self.ifg] = fw / 2
        idx = np.array([0], dtype=np.int32)
        self.av.set_dof_positions(q.reshape(1, -1).astype(np.float32), idx)
        self.av.set_dof_velocities(np.zeros((1, len(q)), np.float32), idx)
        self.tgt = q.copy(); self._apply(); self.odo_reset()

    # ---- state
    def q(self): return self.av.get_dof_positions().reshape(-1).copy()
    def base_pose(self):
        q = self.q(); return float(q[self.ib[0]]), float(q[self.ib[1]]), float(q[self.ib[2]])
    def link_pose(self, name):
        tr = self.av.get_link_transforms().reshape(-1, 7)[self.links.index(name)]
        return tr[:3].copy(), tr[3:].copy()          # pos, quat xyzw
    def tcp_world(self):
        x, y, th = self.base_pose(); T = fk(self.q()[self.ia])
        Tb = np.eye(4); Tb[:3, :3] = Rz(th); Tb[:3, 3] = [x, y, 0]
        Tm = np.eye(4); Tm[:3, 3] = ARM_MOUNT
        return Tb @ Tm @ T

    def _apply(self):
        self.av.set_dof_position_targets(self.tgt.reshape(1, -1).astype(np.float32), np.array([0], dtype=np.int32))

    def step(self, n=1):
        for _ in range(n):
            self._apply(); self.step_fn(); self.sim_t += self.dt
            self._odo_update()
            if self.on_step: self.on_step()

    # ---- odometry (D1/D13): integrates the TRUE base motion with a distance-scale error and a heading drift per metre
    def odo_reset(self, pose=None):
        self.odo = list(pose) if pose is not None else list(self.base_pose()); self._prev_true = self.base_pose(); self.odo_path_m = 0.0
    def _odo_update(self):
        if self.odo is None: self.odo_reset(); return
        x, y, th = self.base_pose(); px, py, pth = self._prev_true
        dx, dy = x - px, y - py; ds = math.hypot(dx, dy); dth = math.atan2(math.sin(th - pth), math.cos(th - pth))
        e = self.odo[2] - pth; c, s_ = math.cos(e), math.sin(e); k = 1.0 + self.odo_s
        self.odo[0] += k * (c * dx - s_ * dy); self.odo[1] += k * (s_ * dx + c * dy); self.odo[2] += dth + self.odo_h * ds
        self.odo_path_m += ds; self._prev_true = (x, y, th)
    def odom_pose(self):
        if self.odo is None: self.odo_reset()
        return tuple(self.odo)
    def odo_to_true(self, xg, yg, thg):
        """map-frame goal as the robot believes it (odometry frame) -> where the base really goes"""
        x, y, th = self.base_pose(); xo, yo, tho = self.odom_pose(); e = tho - th; k = 1.0 + self.odo_s
        dxo, dyo = xg - xo, yg - yo; c, s_ = math.cos(-e), math.sin(-e)
        dx, dy = (c * dxo - s_ * dyo) / k, (s_ * dxo + c * dyo) / k
        L = math.hypot(dx, dy)
        return x + dx, y + dy, thg - e - self.odo_h * L
    def true_to_odo(self, xt, yt, tht):
        """inverse of odo_to_true (oracle helper: a perfectly re-localised command)"""
        x, y, th = self.base_pose(); xo, yo, tho = self.odom_pose(); e = tho - th; k = 1.0 + self.odo_s
        dx, dy = xt - x, yt - y; L = math.hypot(dx, dy); c, s_ = math.cos(e), math.sin(e)
        return xo + k * (c * dx - s_ * dy), yo + k * (s_ * dx + c * dy), tht + e + self.odo_h * L

    # ---- actions
    def wait(self, s):
        s = min(20.0, max(0.0, s)); self.step(int(round(s / self.dt))); return {"stop": "done", "t": self.sim_t}

    def drive_to(self, x, y, heading_deg, speed=0.4, accel=0.5, max_s=5.0, frame="odom"):
        """straight-line trapezoid in x,y with the heading turned along the way (D14). The goal is in the map frame as the robot
        BELIEVES it (its drifting odometry, frame='odom'), or the true map frame (frame='true', scene scripts only). Runs at most
        max_s sim seconds, or until stop_check() returns a reason (a person or moving object inside the safety distance); either
        way the base then brakes at the commanded accel (no instant stop), so a gentle accel stops late."""
        speed = min(abs(speed), 0.6); accel = min(max(0.05, abs(accel)), 2.0)
        if frame == "odom": xt, yt, tht = self.odo_to_true(x, y, math.radians(heading_deg))
        else: xt, yt, tht = x, y, math.radians(heading_deg)
        x0, y0, th0 = self.tgt[self.ib[0]], self.tgt[self.ib[1]], self.tgt[self.ib[2]]
        dth = math.atan2(math.sin(tht - th0), math.cos(tht - th0))
        L = math.hypot(xt - x0, yt - y0); ux, uy = ((xt - x0) / L, (yt - y0) / L) if L > 1e-6 else (0, 0)
        Lr = max(L, abs(dth) * 0.3)              # a pure turn runs as 0.3 m of path per radian
        t = 0.0; reason = "arrived"; s = 0.0; v = 0.0; braking = False
        while True:
            if not braking:
                if t >= max_s: reason = "time_limit"; braking = True
                elif self.stop_check:
                    r = self.stop_check()
                    if r: reason = r; braking = True
            rem = Lr - s
            if braking: v = max(0.0, v - accel * self.dt)
            else:
                vstop = math.sqrt(max(0.0, 2 * accel * rem))          # brake to arrive exactly at the goal
                v = min(speed, v + accel * self.dt, vstop)
                if rem < 1e-5 or (v < 1e-4 and rem < 1e-3): s = Lr; v = 0.0
            s = min(Lr, s + v * self.dt); t += self.dt
            f = min(1.0, s / Lr) if Lr > 0 else 1.0
            self.tgt[self.ib[0]] = x0 + ux * L * f; self.tgt[self.ib[1]] = y0 + uy * L * f; self.tgt[self.ib[2]] = th0 + dth * f
            self.step()
            if v <= 0.0 and (braking or s >= Lr - 1e-6): break
            if t > max_s + 30: reason = "internal_timeout"; break
        self.step(int(0.3 / self.dt))
        bx, by, bth = self.odom_pose()
        return {"stop": reason, "odom_pose": (round(bx, 3), round(by, 3), round(math.degrees(bth), 1)), "t": round(self.sim_t, 2)}

    def gripper(self, width_m, force_n=None, settle_s=0.6):
        """width in m (0..0.08), force in N: the finger drives saturate at force_n each, so closing on an object
        squeezes with that force (Franka hand: 70 N continuous, 140 N peak)."""
        w = float(np.clip(width_m / 2, 0.0, 0.04))
        if os.environ.get("T1_OPEN_DEFAULT", "0") == "1":      # contract: force_n caps the squeeze when CLOSING; an opening command (target wider than now) runs on the build-default finger drive again (a pinned finger needed more than the last grasp's 140 N cap to open, 2026-10-29)
            if not hasattr(self, "_fdrive0"):
                self._fdrive0 = (self.av.get_dof_max_forces().reshape(1, -1).copy(), self.av.get_dof_stiffnesses().reshape(1, -1).copy(), self.av.get_dof_dampings().reshape(1, -1).copy())
            if force_n is None and w > float(np.mean([self.q()[i] for i in self.ifg])) + 1e-4:
                mf0, st0, dp0 = self._fdrive0; mf = self.av.get_dof_max_forces().reshape(1, -1).copy(); st = self.av.get_dof_stiffnesses().reshape(1, -1).copy(); dp = self.av.get_dof_dampings().reshape(1, -1).copy()
                for i in self.ifg: mf[0, i] = mf0[0, i]; st[0, i] = st0[0, i]; dp[0, i] = dp0[0, i]
                self.av.set_dof_max_forces(mf.astype(np.float32), np.array([0], dtype=np.int32)); self.av.set_dof_stiffnesses(st.astype(np.float32), np.array([0], dtype=np.int32)); self.av.set_dof_dampings(dp.astype(np.float32), np.array([0], dtype=np.int32))
        if force_n is not None:
            mf = self.av.get_dof_max_forces().reshape(1, -1).copy()
            for i in self.ifg: mf[0, i] = float(force_n)
            self.av.set_dof_max_forces(mf.astype(np.float32), np.array([0], dtype=np.int32))
            st = self.av.get_dof_stiffnesses().reshape(1, -1).copy()
            for i in self.ifg: st[0, i] = 2.0e4      # 2 cm squeeze past contact = 400 N demand -> saturates at force_n
            self.av.set_dof_stiffnesses(st.astype(np.float32), np.array([0], dtype=np.int32))
            dp = self.av.get_dof_dampings().reshape(1, -1).copy()
            for i in self.ifg: dp[0, i] = 2.0e2
            self.av.set_dof_dampings(dp.astype(np.float32), np.array([0], dtype=np.int32))
        if self.grip_speed:        # velocity-limited fingers (a real hand moves its fingers at a set speed; grip_speed = width speed, m/s)
            q0 = np.array([self.q()[i] for i in self.ifg]); n = max(1, int(math.ceil(float(np.max(np.abs(w - q0))) / (self.grip_speed / 2) / self.dt)))
            for k in range(1, n + 1):
                for j, i in enumerate(self.ifg): self.tgt[i] = q0[j] + (w - q0[j]) * k / n
                self.step()
        for i in self.ifg: self.tgt[i] = w
        self.step(int(settle_s / self.dt))
    def finger_width(self):
        q = self.q(); return float(q[self.ifg[0]] + q[self.ifg[1]])

    def move_joints(self, qa, dur=None, vmax=0.8):
        q0 = self.tgt[self.ia].copy(); dq = np.asarray(qa) - q0
        dur = dur or max(0.6, float(np.max(np.abs(dq))) / vmax * 1.5)
        n = int(dur / self.dt)
        for k in range(1, n + 1):
            s = k / n; s = s * s * (3 - 2 * s)
            self.tgt[self.ia] = q0 + dq * s; self.step()
        self.step(int(0.3 / self.dt))

    def move_to(self, p_world, R_world, gripper_w=None, frame="map", dur=None, line=False):
        """TCP to p (m) with orientation R (3x3) in the map frame (or base frame). Local IK; returns ik residuals."""
        x, y, th = self.odom_pose() if frame == "map" else self.base_pose()     # 'map': the robot's own (drifting) idea of where it is
        if frame in ("map", "true"):
            Rb = Rz(th); pb = Rb.T @ (np.asarray(p_world) - np.array([x, y, 0])); Rbl = Rb.T @ R_world
        else: pb, Rbl = np.asarray(p_world, float), R_world
        p0 = pb - ARM_MOUNT
        qs, pe, re, ok = ik(p0, Rbl, self.tgt[self.ia])
        if not ok:   # retry from fixed seeds; keep the converged solution nearest the current joints
            cur = self.tgt[self.ia]; best = None
            for sd in SEEDS:
                q2, pe2, re2, ok2 = ik(p0, Rbl, sd, iters=250)
                key = (0 if ok2 else 1, float(np.linalg.norm(q2 - cur)) if ok2 else pe2 + re2)
                if best is None or key < best[0]: best = (key, q2, pe2, re2, ok2)
            if best[4] or best[2] + best[3] < pe + re: qs, pe, re, ok = best[1:]
        if gripper_w is not None: self.gripper(gripper_w, settle_s=0.0)
        if line:     # Cartesian straight line: IK chained along waypoints (joint interpolation can swing the fingers through things)
            T0 = fk(self.tgt[self.ia]); pa = T0[:3, 3]; n = max(4, int(np.linalg.norm(p0 - pa) / 0.01))
            qc = self.tgt[self.ia].copy(); traj = []
            for k in range(1, n + 1):
                qc, _, _, _ = ik(pa + (p0 - pa) * k / n, Rbl, qc, iters=60); traj.append(qc.copy())
            dur = dur or max(0.8, np.linalg.norm(p0 - pa) / 0.08)
            m = int(dur / self.dt)
            if SMOOTH_LINE:      # joint targets interpolate between the 1 cm waypoints (a real arm's Cartesian line is continuous)
                knots = [self.tgt[self.ia].copy()] + traj
                for k in range(1, m + 1):
                    f = k / m; f = f * f * (3 - 2 * f); sf = f * n; i = min(n - 1, int(sf)); fr = sf - i
                    self.tgt[self.ia] = knots[i] + fr * (knots[i + 1] - knots[i]); self.step()
            else:
              for k in range(1, m + 1):
                f = k / m; f = f * f * (3 - 2 * f); i = min(n - 1, int(f * n)); 
                self.tgt[self.ia] = traj[i]; self.step()
            self.step(int(0.3 / self.dt))
            T = fk(self.tgt[self.ia]); pe = float(np.linalg.norm(T[:3, 3] - p0)); re = float(np.linalg.norm(rot_err(T[:3, :3], Rbl))); ok = pe < 3e-3
        else:
            self.move_joints(qs, dur)
        return {"ik_ok": ok, "ik_pos_err_m": round(pe, 4), "ik_rot_err_rad": round(re, 4), "t": round(self.sim_t, 2)}


def author_initial_pose(st, root, qa=Q_HOME, base=(0.0, 0.0, 0.0), fw=0.08):
    """write the start joint state + drive targets into USD before play, so the arm never spawns in its stretched default pose"""
    from pxr import PhysxSchema, UsdPhysics
    vals = {"dummy_base_prismatic_x_joint": ("linear", base[0]), "dummy_base_prismatic_y_joint": ("linear", base[1]),
            "dummy_base_revolute_z_joint": ("angular", math.degrees(base[2])),
            "panda_finger_joint1": ("linear", fw / 2), "panda_finger_joint2": ("linear", fw / 2)}
    for i, v in enumerate(qa): vals[f"panda_joint{i + 1}"] = ("angular", math.degrees(v))
    from pxr import Usd
    for p in Usd.PrimRange(st.GetPrimAtPath(root)):
        if p.GetName() in vals:
            kind, v = vals[p.GetName()]
            js = PhysxSchema.JointStateAPI.Apply(p, kind); js.CreatePositionAttr().Set(float(v)); js.CreateVelocityAttr().Set(0.0)
            UsdPhysics.DriveAPI.Apply(p, kind).CreateTargetPositionAttr().Set(float(v))


def _ready():
    best = None
    for sd in [np.array([0, -0.5, 0, -1.2, 0, 1.4, 0.785])] + SEEDS:
        q, pe, re, ok = ik(np.array([0.60, 0, 1.20]) - ARM_MOUNT, R_pitch(0.0, 45), sd, iters=400)
        if ok and (best is None or abs(q[0]) + abs(q[2]) < abs(best[0]) + abs(best[2])): best = q
    return best


Q_READY = _ready()      # carry / ready pose: TCP 0.60 m ahead of the base centre, 1.20 m high, pitched 45 deg down
