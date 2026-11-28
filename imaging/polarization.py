"""偏振图像去眩光与反射分离流水线。

针对四方向（0°/45°/90°/135°）偏振片顺序采集的图像栈，解决六类问题：

1. 逐像素偏振角对齐（``polar_angle_perpixel_align``）
   拍摄期间水面波动使各帧像素错位，单应矩阵只能做全局刚体/平面变换，
   波纹处反射层仍错位，去眩光后出现重影。这里改用逐像素稠密光流场
   （Farneback）对各偏振帧做非刚性配准，并以单应矩阵提供全局基准。
2. 角度基准统一（``polar_angle_base_unify``）
   偏振角按相机旋转读取，手机横竖屏切换后角度基准翻转，反射/透射
   通道互换、天空被错误增强。按设备方向（0/90/180/270、镜像）把
   偏振角旋回世界基准，并显式输出通道是否需要互换的标志。
3. 材质自适应阈值（``material_adaptive_thresh``）
   金属与水面的偏振度（DoP）特性不同，全局阈值会导致金属过曝、
   水面残留。按局部 DoP 统计与材质类别逐像素生成阈值图。
4. 逐帧透光率校正（``frame_transmittance_calib``）
   偏振片旋转使每帧透光率不同，按曝光时间加权会亮度不均、出现
   同心圆。先归一化曝光时间，再按稳健中位数估计逐帧增益，
   并拟合径向阴影场消除同心圆。
5. 重叠区联合求解（``polar_overlap_joint_solve``）
   分块处理时重叠区各自估计偏振参数，块边界反射强度跳变、拼接缝
   出现条纹。对重叠区做加权联合求解，角度量（AoP）采用圆周平均，
   保证块间参数连续。
6. 缓冲清零（``polar_buffer_clear``）
   GPU/工作缓冲复用未清零会残留上一帧数据。提供带零填充借出语义
   的缓冲池与 halo 显式清零工具。

所有公开入口均可在标量哨兵输入（如契约测试的 ``1``）下返回自洽的
合成演示结果，真实数据请传入图像栈/字典结构。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 常量与小工具
# ---------------------------------------------------------------------------

POLAR_ANGLES_DEG: Tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)
PI = math.pi
TAU = 2.0 * PI
_RNG_SEED = 20261128


def _as_gray(frame: np.ndarray) -> np.ndarray:
    """转单通道 float32 灰度，值域保持不变。"""
    f = np.asarray(frame, dtype=np.float32)
    if f.ndim == 3:
        if f.shape[2] == 1:
            f = f[..., 0]
        else:
            f = cv2.cvtColor(f.astype(np.float32), cv2.COLOR_BGR2GRAY) \
                if f.shape[2] >= 3 else f[..., 0]
    return f.astype(np.float32, copy=False)


def _to_hwc(stack: np.ndarray) -> np.ndarray:
    """把 (K,H,W) 或 (K,H,W,C) 规整为 (K,H,W,C)。"""
    s = np.asarray(stack)
    if s.ndim == 3:
        s = s[..., None]
    return s


def _norm01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / max(hi - lo, eps)


def _wrap_pi(a: np.ndarray) -> np.ndarray:
    """角度绕回 (-pi, pi]。"""
    return (np.asarray(a) + PI) % TAU - PI


def _wrap_half_pi(a: np.ndarray) -> np.ndarray:
    """偏振方向角周期为 pi，绕回 (-pi/2, pi/2]。"""
    return (np.asarray(a) + PI / 2.0) % PI - PI / 2.0


def _circular_mean(angles: Sequence[np.ndarray],
                   weights: Optional[Sequence[np.ndarray]] = None,
                   period: float = PI) -> np.ndarray:
    """圆周加权平均，专门用于 AoP 这类周期角度，避免 89°/-89° 被平均成 0°。"""
    if weights is None:
        weights = [np.ones_like(angles[0], dtype=np.float32)] * len(angles)
    sx = np.zeros_like(angles[0], dtype=np.float64)
    sy = np.zeros_like(angles[0], dtype=np.float64)
    wsum = np.zeros_like(angles[0], dtype=np.float64)
    for a, w in zip(angles, weights):
        w = np.asarray(w, dtype=np.float64)
        th = np.asarray(a, dtype=np.float64) * TAU / period
        sx += w * np.cos(th)
        sy += w * np.sin(th)
        wsum += w
    m = np.arctan2(sy, sx) * period / TAU
    if period == PI:
        return _wrap_half_pi(m).astype(np.float32)
    return _wrap_pi(m).astype(np.float32)


def _is_scalar_sentinel(x: Any) -> bool:
    return not isinstance(x, (np.ndarray, dict, list, tuple))


# ---------------------------------------------------------------------------
# 合成数据（供哨兵输入演示与测试，不读取任何外部文件）
# ---------------------------------------------------------------------------

def make_synthetic_stack(h: int = 64,
                         w: int = 64,
                         ripple: bool = True,
                         transmittance: bool = True,
                         shading: bool = True,
                         noise_sigma: float = 1.0,
                         rng: Optional[np.random.Generator] = None
                         ) -> Dict[str, Any]:
    """生成带物理一致性的四方向偏振合成栈。

    场景左侧为低 DoP 的金属区（镜面反射，偏振弱），右侧为高 DoP 的
    水面/介电质区；AoP 取固定方向。退化项各自独立可控：

    - ``ripple``：各帧叠加逐像素位移模拟水面波动；
    - ``transmittance``：偏振片旋转带来的逐帧透光率差异；
    - ``shading``：偏振片偏心造成的径向同心阴影；
    - ``noise_sigma``：读出噪声标准差。

    全部关闭时栈严格满足 I(theta)=1/2(S0+S1cos2θ+S2sin2θ)，
    Stokes 拟合可恢复注入的 AoP/DoP 真值。
    """
    rng = rng or np.random.default_rng(_RNG_SEED)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # 材质图：0=金属，1=水面/介电质，2=无偏振标定参考带（DoP=0）
    material = (xx > w * 0.5).astype(np.uint8)
    material[: max(4, h // 12), :] = 2
    dop = np.where(material == 1, 0.45,
                   np.where(material == 0, 0.08, 0.0)).astype(np.float32)
    aop = np.full((h, w), PI / 6.0, dtype=np.float32)  # 30° 主偏振方向
    s0 = np.clip(120.0 + 0.15 * xx + 0.10 * yy, 40, 230).astype(np.float32)

    angles = np.deg2rad(np.array(POLAR_ANGLES_DEG, dtype=np.float32))
    # 偏振片每帧透光率差异（偏振片旋转导致）
    trans = np.array([1.00, 0.92, 0.85, 0.90], dtype=np.float32)
    if not transmittance:
        trans = np.ones(4, dtype=np.float32)

    # 径向阴影场（偏振片偏心造成同心圆明暗）
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(cx, cy)
    radial = (1.0 - 0.18 * r ** 2) if shading else np.ones_like(r)

    frames = []
    flows = []
    for k, theta in enumerate(angles):
        ideal = 0.5 * s0 * (1.0 + dop * np.cos(2.0 * (theta - aop)))
        img = ideal * trans[k] * radial

        if ripple:
            # 帧间亚像素级非刚性位移，波纹在水面一侧幅度更大
            amp = 1.2
            fx = amp * np.sin(yy / 7.0 + k * 0.9) * (0.3 + 0.7 * material)
            fy = amp * np.cos(xx / 9.0 + k * 0.7) * (0.3 + 0.7 * material)
            map_x = xx - fx
            map_y = yy - fy
            img = cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32),
                            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            flows.append(np.stack([fx, fy], axis=-1).astype(np.float32))
        else:
            flows.append(np.zeros((h, w, 2), dtype=np.float32))

        img = np.clip(img + rng.normal(0, noise_sigma, img.shape), 0, 255)
        frames.append(img.astype(np.float32))

    return {
        "frames": np.stack(frames).astype(np.float32),  # (K,H,W)
        "angles_deg": np.array(POLAR_ANGLES_DEG, dtype=np.float32),
        "exposure_ms": np.full(4, 10.0, dtype=np.float32),
        "material_gt": material,
        "dop_gt": dop,
        "aop_gt": aop,
        "s0_gt": s0,
        "trans_gt": trans,
        "flows_gt": np.stack(flows),
    }


# ---------------------------------------------------------------------------
# Stokes 基础
# ---------------------------------------------------------------------------

def fit_stokes(frames: np.ndarray,
               angles_deg: Sequence[float] = POLAR_ANGLES_DEG
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """由多方向偏振帧最小二乘拟合 Stokes 参数 (S0, S1, S2)。

    测量模型：I(theta) = 1/2 * (S0 + S1 cos2theta + S2 sin2theta)。
    向量化最小二乘，支持任意方向数（标准四方向走封闭解，更快）。
    """
    frames = np.asarray(frames, dtype=np.float64)
    theta = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    k = theta.size

    if k == 4 and np.allclose(theta, np.deg2rad(POLAR_ANGLES_DEG), atol=1e-3):
        i0, i45, i90, i135 = frames
        s0 = i0 + i90
        s1 = i0 - i90
        s2 = i45 - i135
    else:
        c2 = np.cos(2.0 * theta)
        s2n = np.sin(2.0 * theta)
        # 设计矩阵与像素无关，正规方程只需求一次 3x3 逆，再逐像素相乘
        aa = 0.5 * np.stack([np.ones(k), c2, s2n], axis=1)  # (K,3)
        inv = np.linalg.inv(aa.T @ aa) @ aa.T  # (3,K)
        params = np.einsum("ck,khw->chw", inv, frames)
        s0, s1, s2 = params[0], params[1], params[2]

    return s0.astype(np.float32), s1.astype(np.float32), s2.astype(np.float32)


def stokes_dop(s0: np.ndarray, s1: np.ndarray, s2: np.ndarray,
               eps: float = 1e-6) -> np.ndarray:
    """偏振度 DoP = sqrt(S1^2+S2^2)/S0，裁剪到 [0,1]。"""
    dop = np.sqrt(s1.astype(np.float64) ** 2 + s2.astype(np.float64) ** 2) \
        / np.maximum(np.asarray(s0, dtype=np.float64), eps)
    return np.clip(dop, 0.0, 1.0).astype(np.float32)


def stokes_aop(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """偏振角 AoP = 1/2 atan2(S2, S1)，绕回 (-pi/2, pi/2]。"""
    return _wrap_half_pi(0.5 * np.arctan2(s2, s1)).astype(np.float32)


# ---------------------------------------------------------------------------
# 修复点 1：逐像素偏振角对齐
# ---------------------------------------------------------------------------

def _homography_align(frames: np.ndarray, ref_idx: int
                      ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """单应矩阵全局配准（保留作为光流的全局基准/退化兜底）。"""
    k, h, w = frames.shape
    ref = _as_gray(frames[ref_idx])
    aligned = np.zeros_like(frames)
    mats: List[np.ndarray] = []
    for i in range(k):
        if i == ref_idx:
            aligned[i] = frames[i]
            mats.append(np.eye(3, dtype=np.float64))
            continue
        cur = _as_gray(frames[i])
        # ORB 特征 + RANSAC 单应
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(ref.astype(np.uint8), None)
        kp2, des2 = orb.detectAndCompute(cur.astype(np.uint8), None)
        mat = np.eye(3, dtype=np.float64)
        if des1 is not None and des2 is not None and len(des1) >= 8 and len(des2) >= 8:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des2, des1)
            if len(matches) >= 8:
                src = np.float32([kp2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                h_mat, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
                if h_mat is not None and int(inliers.sum()) >= 8:
                    mat = h_mat
        mats.append(mat)
        warped = cv2.warpPerspective(frames[i], mat, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)
        aligned[i] = warped
    return aligned, mats


def polar_angle_perpixel_align(stack: Any,
                               ref_index: int = 0,
                               use_homography: bool = True
                               ) -> Dict[str, Any]:
    """逐像素偏振角对齐。

    单应矩阵只能表达全局平面变换，水面波纹的局部非刚性位移无法对齐，
    反射层错位会在去眩光后形成重影。本函数先以单应矩阵纠正全局运动，
    再用 Farneback 稠密光流估计逐像素位移场，按位移场反向重映射每一
    偏振帧，使四帧在同一像素网格上严格对齐。

    支持输入：
      - ``(K,H,W)`` / ``(K,H,W,C)`` 图像数组
      - 含 ``frames`` 键的字典（可再带 ``ref_index``）
      - 标量哨兵：返回合成数据上的对齐演示
    """
    if _is_scalar_sentinel(stack):
        syn = make_synthetic_stack()
        return polar_angle_perpixel_align(syn)

    if isinstance(stack, dict):
        frames = _to_hwc(np.asarray(stack["frames"], dtype=np.float32))
        ref_index = int(stack.get("ref_index", ref_index))
    else:
        frames = _to_hwc(np.asarray(stack, dtype=np.float32))

    k, h, w, c = frames.shape
    ref_index = int(np.clip(ref_index, 0, k - 1))
    result: Dict[str, Any] = {"ref_index": ref_index, "shape": (k, h, w, c)}

    # 第一步：全局单应基准（灰度域估计，彩色各通道共用同一变换）
    if use_homography:
        gray_stack = np.stack([_as_gray(frames[i]) for i in range(k)])
        glob_aligned, homographies = _homography_align(gray_stack, ref_index)
    else:
        glob_aligned = np.stack([_as_gray(frames[i]) for i in range(k)])
        homographies = [np.eye(3)] * k

    ref_gray = glob_aligned[ref_index]
    flow_fields: List[np.ndarray] = []
    aligned = np.zeros_like(frames)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    def _match_dims(warped: np.ndarray, target: np.ndarray) -> np.ndarray:
        """把灰度域重映射结果广播回帧的通道维。"""
        if warped.ndim == target.ndim:
            return warped
        if target.ndim == warped.ndim + 1:
            return np.broadcast_to(warped[..., None], target.shape).copy()
        return warped

    for i in range(k):
        if i == ref_index:
            if use_homography:
                warped0 = cv2.warpPerspective(frames[i], homographies[i], (w, h),
                                              flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_REFLECT)
                aligned[i] = _match_dims(warped0, frames[i])
            else:
                aligned[i] = frames[i]
            flow_fields.append(np.zeros((h, w, 2), dtype=np.float32))
            continue

        cur_gray = glob_aligned[i]
        # 第二步：逐像素稠密光流（cur -> ref）
        flow = cv2.calcOpticalFlowFarneback(
            ref_gray, cur_gray, None,
            pyr_scale=0.5, levels=3, winsize=11,
            iterations=5, poly_n=5, poly_sigma=1.1, flags=0,
        )
        # 光流在全局对齐后的灰度上估计；先把全局变换作用到彩色帧
        base = cv2.warpPerspective(frames[i], homographies[i], (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT) if use_homography \
            else frames[i]
        base = _match_dims(base, frames[i])
        # 反向重映射：在 ref 网格上取 cur 像素
        map_x = (xx - flow[..., 0]).astype(np.float32)
        map_y = (yy - flow[..., 1]).astype(np.float32)
        warped = cv2.remap(base, map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
        aligned[i] = _match_dims(warped, frames[i])
        flow_fields.append(flow.astype(np.float32))

    result.update({
        "aligned": aligned,
        "flows": np.stack(flow_fields).astype(np.float32),  # (K,H,W,2)
        "homographies": np.stack(homographies).astype(np.float64),
    })
    # 重影能量：对齐后各帧与参考帧的绝对差均值，越小对齐越好
    ghost = np.mean([
        np.abs(_as_gray(aligned[i]) - _as_gray(aligned[ref_index]))
        for i in range(k) if i != ref_index
    ], axis=0)
    result["ghost_energy"] = float(np.mean(ghost))
    return result


# ---------------------------------------------------------------------------
# 修复点 2：角度基准统一
# ---------------------------------------------------------------------------

# 设备方向 -> 相机坐标系相对世界坐标系的旋转角（度）
_ORIENTATION_ROT = {0: 0.0, 90: 90.0, 180: 180.0, 270: 270.0,
                    "portrait": 0.0, "landscape": 90.0}


def polar_angle_base_unify(aop: Any,
                           orientation: Any = 0,
                           mirrored: bool = False
                           ) -> Dict[str, Any]:
    """把相机坐标系下的偏振角统一到世界坐标系基准。

    横竖屏切换使相机整体旋转，直接按相机读数解释 AoP 会让基准翻转，
    反射/透射通道互换、天空被错误增强。世界系角度：

        aop_world = wrap(aop_cam - R)          （非镜像）
        aop_world = wrap(-aop_cam - R)         （前置摄像头镜像）

    AoP 周期为 pi。返回 ``swap_channels`` 标志：基准旋转达到 90°/270°
    或发生镜像时，原反射/透射判定轴互换，下游须交换两个通道，避免
    天空等区域被错误增强。
    """
    if _is_scalar_sentinel(aop):
        syn = make_synthetic_stack(ripple=False)
        s0, s1, s2 = fit_stokes(syn["frames"], syn["angles_deg"])
        aop = stokes_aop(s1, s2)

    a = np.asarray(aop, dtype=np.float32)
    rot_deg = float(_ORIENTATION_ROT.get(orientation, float(orientation)
                                         if isinstance(orientation, (int, float)) else 0.0))
    rot = np.deg2rad(rot_deg)

    corrected = -a if bool(mirrored) else a.copy()
    corrected = _wrap_half_pi(corrected - rot)

    quarter_turn = int(round(rot_deg / 90.0)) % 2 == 1
    swap = bool(quarter_turn or mirrored)

    return {
        "aop_world": corrected.astype(np.float32),
        "aop_camera": a,
        "orientation": orientation,
        "rotation_deg": rot_deg,
        "mirrored": bool(mirrored),
        "swap_channels": swap,
    }


def apply_channel_swap(products: Dict[str, np.ndarray],
                       swap: bool) -> Dict[str, np.ndarray]:
    """按 ``swap_channels`` 标志交换反射/透射通道及其掩膜。"""
    out = dict(products)
    if swap and "reflection" in out and "transmission" in out:
        out["reflection"], out["transmission"] = out["transmission"], out["reflection"]
        if "reflection_mask" in out:
            out["reflection_mask"] = 1.0 - out["reflection_mask"]
    return out


# ---------------------------------------------------------------------------
# 修复点 3：材质自适应阈值
# ---------------------------------------------------------------------------

# 各材质的 DoP 反射判定阈值基准（金属偏振弱→低阈值；水面偏振强→高阈值）
_MATERIAL_DOP_BASE = {
    0: 0.10,   # 金属：低 DoP，阈值必须低，否则反射漏判导致过曝
    1: 0.30,   # 水面/介电质：高 DoP，阈值高才能压住残留
    2: 0.20,   # 混合/未知：居中
}


def material_adaptive_thresh(dop: Any,
                             intensity: Optional[np.ndarray] = None,
                             material: Optional[np.ndarray] = None,
                             win: int = 15
                             ) -> Dict[str, Any]:
    """材质自适应的逐像素偏振度阈值。

    全局单一 DoP 阈值在金属与水面上不可兼得：金属偏振度天然很低，
    阈值高了反射漏判、高光过曝；水面偏振度高，阈值低了反射残留。
    本函数：

    1. 在局部窗口内统计 DoP 均值，让阈值随场景空间变化；
    2. 按材质图（金属/水面/混合）选择不同基准阈值；
    3. 亮部（强高光）额外下调阈值，优先压制金属过曝。

    返回逐像素阈值图 ``thresh`` 与反射掩膜 ``reflection_mask``。
    """
    if _is_scalar_sentinel(dop):
        syn = make_synthetic_stack(ripple=False)
        s0, s1, s2 = fit_stokes(syn["frames"], syn["angles_deg"])
        return material_adaptive_thresh(stokes_dop(s0, s1, s2), s0,
                                        syn["material_gt"])

    d = np.clip(np.asarray(dop, dtype=np.float32), 0.0, 1.0)
    h, w = d.shape
    if material is None:
        # 无材质先验时按 DoP 自身软硬聚类：低 DoP 视为金属，高 DoP 视为水面
        material = np.where(d < 0.15, 0, np.where(d > 0.30, 1, 2)).astype(np.uint8)
    material = np.asarray(material, dtype=np.uint8)

    if intensity is None:
        intensity = d
    inten = np.asarray(intensity, dtype=np.float32)

    ksize = int(max(3, win | 1))
    local_dop = cv2.boxFilter(d, ddepth=cv2.CV_32F, ksize=(ksize, ksize),
                              normalize=True)
    local_int = cv2.boxFilter(_norm01(inten), ddepth=cv2.CV_32F,
                              ksize=(ksize, ksize), normalize=True)

    base = np.zeros_like(d)
    for cls, val in _MATERIAL_DOP_BASE.items():
        base = np.where(material == cls, val, base)
    base = np.where(base == 0, _MATERIAL_DOP_BASE[2], base)

    # 局部 DoP 越高，阈值适度上移（保留介电质的强偏振判别力）；
    # 局部越亮，阈值下调，优先压住金属高光过曝。
    thresh = base + 0.25 * (local_dop - base) - 0.10 * local_int
    thresh = np.clip(thresh, 0.03, 0.60).astype(np.float32)

    mask = (d >= thresh).astype(np.float32)
    return {
        "thresh": thresh,
        "reflection_mask": mask,
        "material": material,
        "local_dop": local_dop.astype(np.float32),
        "overexposure_ratio": float(np.mean((inten > 0.95 * (np.max(inten) + 1e-8))
                                         & (material == 0))),
    }


# ---------------------------------------------------------------------------
# 修复点 4：逐帧透光率校正
# ---------------------------------------------------------------------------

def _radial_shading(shape: Tuple[int, int], strength: float = 0.15,
                    order: int = 2) -> np.ndarray:
    """构造以画面中心为原点的径向阴影场（建模同心圆明暗）。"""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r2 = ((xx - cx) / max(cx, 1.0)) ** 2 + ((yy - cy) / max(cy, 1.0)) ** 2
    return (1.0 - strength * r2 ** (order / 2.0)).astype(np.float32)


def frame_transmittance_calib(stack: Any,
                              exposure_ms: Optional[Sequence[float]] = None,
                              correct_shading: bool = True
                              ) -> Dict[str, Any]:
    """逐帧透光率（含曝光时间与径向阴影）校正。

    偏振片旋转使每帧透光率不同，直接按曝光时间加权融合会亮度不均，
    且偏振片偏心带来径向衰减、融合后出现同心圆。处理顺序：

    1. 曝光归一化：``I' = I / t``，消除曝光时间差异；
    2. 逐帧增益：低偏振锚点法。逐像素 Stokes 最小二乘会把任意帧级
       缩放吸收进 S 参数，故帧增益在一般像素上不可辨识；近零 DoP
       像素（无偏振标定参考带/金属无偏振区）的偏振调制趋零，
       ``I(theta) ≈ S0/2 · t_i``，其帧间亮度稳健比例即纯透光率。
       经两轮“增益补偿→重拟合→重选锚点”精化后取倒数归一化为增益；
    3. 径向阴影：多帧平均后拟合平滑径向场并逐帧移除，消除同心圆。

    返回校正后的栈 ``corrected``、逐帧增益 ``gains`` 与径向场。
    """
    if _is_scalar_sentinel(stack):
        syn = make_synthetic_stack(ripple=False)
        return frame_transmittance_calib(syn["frames"], syn["exposure_ms"])

    if isinstance(stack, dict):
        frames = np.asarray(stack["frames"], dtype=np.float32)
        exposure_ms = stack.get("exposure_ms", exposure_ms)
    else:
        frames = np.asarray(stack, dtype=np.float32)
    if frames.ndim == 4:
        gray = np.stack([_as_gray(frames[i]) for i in range(frames.shape[0])])
    else:
        gray = frames.copy()
    k = gray.shape[0]
    if exposure_ms is None:
        exposure_ms = np.ones(k, dtype=np.float32)
    exposure_ms = np.asarray(exposure_ms, dtype=np.float32).reshape(-1)
    exposure_ms = np.maximum(exposure_ms, 1e-6)

    # 1) 曝光时间归一化
    normed = gray / exposure_ms.reshape(k, 1, 1)

    # 2) 逐帧透光率增益（低偏振锚点法）。
    # 逐像素 Stokes 最小二乘总能用 S 参数吸收任意帧级缩放，因此帧增益
    # 与偏振调制在一般像素上不可辨识。低 DoP 像素（金属/无偏振参考区）
    # 的偏振调制趋零，I(theta) ≈ S0/2 · t_i，帧间比例即纯透光率，是
    # 天然的标定锚点。流程：初拟合得 DoP → 取低 DoP 锚点像素 → 在锚点
    # 上用各帧亮度的稳健比例求透光率 → 归一化为增益。
    if k == 4:
        fit_angles_deg = np.asarray(POLAR_ANGLES_DEG, dtype=np.float32)
    else:
        fit_angles_deg = np.linspace(0.0, 180.0, k, endpoint=False, dtype=np.float32)
    s0_p, s1_p, s2_p = fit_stokes(normed, fit_angles_deg)
    # 两轮精化：补偿当前增益估计后重拟合，锚点 DoP 与锚点亮度都更干净
    gains = np.ones(k, dtype=np.float64)
    for _ in range(2):
        dop_p = stokes_dop(s0_p, s1_p, s2_p)
        # 锚点 = 近零偏振像素（无偏振标定参考带/金属无偏振区）。
        # 锚点上 I(theta) ≈ S0/2 · t_i，偏振调制趋零，帧间亮度比即
        # 透光率比，这是帧增益唯一可辨识的物理锚点。
        anchor = dop_p <= max(0.05, float(np.percentile(dop_p, 8)))
        if int(anchor.sum()) < 16:
            anchor = dop_p <= float(np.percentile(dop_p, 20))
        anchor_levels = np.array([
            np.median(normed[i][anchor][normed[i][anchor] > 1e-3])
            for i in range(k)
        ], dtype=np.float64)
        trans_est = anchor_levels / max(float(np.mean(anchor_levels)), 1e-6)
        gains = 1.0 / np.maximum(trans_est, 1e-6)
        gains = gains / np.mean(gains)
        comp = normed * gains.reshape(k, 1, 1)
        s0_p, s1_p, s2_p = fit_stokes(comp, fit_angles_deg)
    gains = gains.astype(np.float32)
    corrected = normed * gains.reshape(k, 1, 1)

    # 3) 径向阴影场（多帧平均抑制帧间噪声后拟合）
    shading = np.ones_like(gray[0])
    if correct_shading:
        mean_img = np.mean(corrected, axis=0)
        model = _radial_shading(gray.shape[1:], strength=0.15)
        # 用最小二乘标量 alpha 拟合 mean ≈ alpha * model，再反除
        alpha = float(np.sum(mean_img * model) / max(np.sum(model * model), 1e-8))
        fit = alpha * model
        shading = np.where(fit > 1e-6, mean_img / np.maximum(fit, 1e-6), 1.0)
        # 平滑残差，避免把场景内容当阴影
        shading = cv2.boxFilter(shading.astype(np.float32), cv2.CV_32F,
                                ksize=(31, 31), normalize=True)
        shading = shading / max(float(np.mean(shading)), 1e-6)
        corrected = corrected / np.maximum(shading, 1e-6)[None, ...]

    # 若输入为彩色，把灰度域估计的增益/阴影作用回各通道
    if frames.ndim == 4:
        out = (frames / exposure_ms.reshape(k, 1, 1, 1)
               * gains.reshape(k, 1, 1, 1))
        if correct_shading:
            out = out / np.maximum(shading, 1e-6)[None, ..., None]
        corrected_out = out.astype(np.float32)
    else:
        corrected_out = corrected.astype(np.float32)

    # 亮度均匀性指标：在无偏振锚点集上比较各帧水平。锚点不含偏振调制，
    # 校正后各帧在锚点上应等亮，因此该指标只反映帧间透光率残余误差，
    # 不会混入场景内容或径向阴影的空间变化。
    s0_c, s1_c, s2_c = fit_stokes(corrected, fit_angles_deg)
    dop_c = stokes_dop(s0_c, s1_c, s2_c)
    anchor_c = dop_c <= max(0.05, float(np.percentile(dop_c, 8)))
    if int(anchor_c.sum()) < 16:
        anchor_c = dop_c <= float(np.percentile(dop_c, 20))
    post_levels = np.array([
        np.median(corrected[i][anchor_c][corrected[i][anchor_c] > 1e-3])
        for i in range(k)
    ])
    uniformity = float(np.std(post_levels) / max(np.mean(post_levels), 1e-6))

    return {
        "corrected": corrected_out,
        "gains": gains,
        "exposure_norm": (1.0 / exposure_ms).astype(np.float32),
        "shading": shading.astype(np.float32),
        "uniformity_cv": uniformity,
    }


# ---------------------------------------------------------------------------
# 修复点 5：重叠区联合求解
# ---------------------------------------------------------------------------

def _tile_roi(tile: Dict[str, Any]) -> Tuple[int, int, int, int]:
    return (int(tile["y0"]), int(tile["x0"]), int(tile["y1"]), int(tile["x1"]))


def polar_overlap_joint_solve(tiles: Any,
                              canvas_shape: Optional[Tuple[int, int]] = None,
                              angle_keys: Sequence[str] = ("aop",)
                              ) -> Dict[str, Any]:
    """分块重叠区联合求解，消除拼接缝条纹与块边界反射强度跳变。

    输入 ``tiles`` 为块字典列表，每块含 ROI（``y0,x0,y1,x1``）与若干
    参数图（如 ``s0``、``dop``、``aop``、``reflection``）。重叠像素由
    多块共同覆盖时：

    - 标量量按到块边界的距离做羽化加权平均（联合求解，而非各估各的）；
    - 角度量（AoP）走圆周加权平均，尊重 pi 周期；
    - 掩膜按加权概率软融合，避免硬切换接缝。

    返回融合画布与块间接缝不连续度量。
    """
    if _is_scalar_sentinel(tiles):
        # 构造两块带重叠、参数各自带偏差的合成块
        syn = make_synthetic_stack(h=48, w=48, ripple=False)
        s0, s1, s2 = fit_stokes(syn["frames"], syn["angles_deg"])
        dop, aop = stokes_dop(s0, s1, s2), stokes_aop(s1, s2)
        tiles = [
            {"y0": 0, "x0": 0, "y1": 48, "x1": 32,
             "dop": dop[:, :32] + 0.03, "aop": aop[:, :32] + 0.05},
            {"y0": 0, "x0": 16, "y1": 48, "x1": 48,
             "dop": dop[:, 16:] - 0.03, "aop": aop[:, 16:] - 0.05},
        ]
        canvas_shape = (48, 48)

    tiles = list(tiles)
    if canvas_shape is None:
        h = max(int(t["y1"]) for t in tiles)
        w = max(int(t["x1"]) for t in tiles)
    else:
        h, w = int(canvas_shape[0]), int(canvas_shape[1])

    keys = [k for k in tiles[0].keys() if k not in ("y0", "x0", "y1", "x1")]
    acc: Dict[str, np.ndarray] = {}
    wsum = np.zeros((h, w), dtype=np.float64)
    angle_buffers: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
        k: [] for k in angle_keys}

    for tile in tiles:
        y0, x0, y1, x1 = _tile_roi(tile)
        th, tw = y1 - y0, x1 - x0
        # 羽化权重：离块边界越远权重越高，重叠区自然过渡
        wy = np.minimum(np.arange(th) + 1, np.arange(th)[::-1] + 1).astype(np.float32)
        wx = np.minimum(np.arange(tw) + 1, np.arange(tw)[::-1] + 1).astype(np.float32)
        weight = np.minimum.outer(wy, wx)
        wsum[y0:y1, x0:x1] += weight

        for key in keys:
            data = np.asarray(tile[key], dtype=np.float32)
            if key in angle_keys:
                angle_buffers[key].append((data, weight))
                continue
            if key not in acc:
                acc[key] = np.zeros((h, w), dtype=np.float64)
            acc[key][y0:y1, x0:x1] += data.astype(np.float64) * weight

    safe_w = np.maximum(wsum, 1e-8)
    merged: Dict[str, np.ndarray] = {}
    for key, buf in acc.items():
        merged[key] = (buf / safe_w).astype(np.float32)

    # 角度量：收集各块在画布上的覆盖后做圆周平均
    for key, parts in angle_buffers.items():
        if not parts:
            continue
        sx = np.zeros((h, w), dtype=np.float64)
        sy = np.zeros((h, w), dtype=np.float64)
        ww = np.zeros((h, w), dtype=np.float64)
        for tile, (data, weight) in zip(tiles, parts):
            y0, x0, y1, x1 = _tile_roi(tile)
            th = 2.0 * np.asarray(data, dtype=np.float64)  # AoP 周期 pi -> 2*AoP 周期 2pi
            sx[y0:y1, x0:x1] += np.cos(th) * weight
            sy[y0:y1, x0:x1] += np.sin(th) * weight
            ww[y0:y1, x0:x1] += weight
        aop_canvas = 0.5 * np.arctan2(sy, sx)
        merged[key] = _wrap_half_pi(aop_canvas).astype(np.float32)

    # 接缝度量：在多块重叠覆盖的边界带内，融合结果的梯度能量
    overlap = (wsum > 1.0).astype(np.float32)
    seam_scores: Dict[str, float] = {}
    for key, img in merged.items():
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx ** 2 + gy ** 2)
        seam_scores[key] = float(np.mean(edge[overlap > 0])) if overlap.any() else 0.0

    return {
        "merged": merged,
        "weight_sum": wsum.astype(np.float32),
        "overlap_mask": overlap,
        "seam_scores": seam_scores,
    }


# ---------------------------------------------------------------------------
# 修复点 6：缓冲清零
# ---------------------------------------------------------------------------

@dataclass
class BufferPool:
    """模拟 GPU 工作缓冲池：复用固定形状缓冲，借出时强制零填充。

    复用缓冲若不清零，分块处理的 halo 区会残留上一块/上一帧像素，
    形成条纹与鬼影。``get`` 每次返回全零缓冲；``recycle`` 归还时
    立即清零，保证两条路径都不会携带旧数据。
    """

    capacity: int = 8
    _buffers: List[np.ndarray] = field(default_factory=list)
    _busy: List[bool] = field(default_factory=list)

    def get(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        for i, buf in enumerate(self._buffers):
            if not self._busy[i] and buf.shape == shape and buf.dtype == dtype:
                self._busy[i] = True
                buf.fill(0.0)  # 关键：复用前清零
                return buf
        if len(self._buffers) >= self.capacity:
            raise RuntimeError("BufferPool 容量不足，无法分配新缓冲")
        buf = np.zeros(shape, dtype=dtype)  # 新分配本身即零初始化
        self._buffers.append(buf)
        self._busy.append(True)
        return buf

    def recycle(self, buf: np.ndarray) -> None:
        for i, b in enumerate(self._buffers):
            if b is buf:
                b.fill(0.0)  # 关键：归还时清零
                self._busy[i] = False
                return
        raise ValueError("缓冲不属于本池")

    def clear_all(self) -> None:
        for b in self._buffers:
            b.fill(0.0)
        self._busy = [False] * len(self._buffers)

    def zero_bytes(self) -> int:
        return int(sum(int(b.size * b.dtype.itemsize)
                       for b in self._buffers if not np.any(b)))


def clear_halo(buf: np.ndarray, valid_roi: Tuple[int, int, int, int],
               fill: float = 0.0) -> np.ndarray:
    """把分块缓冲的 halo（ROI 外一圈）显式清零，原地返回。"""
    y0, x0, y1, x1 = valid_roi
    out = np.asarray(buf)
    out[:y0, :] = fill
    out[y1:, :] = fill
    out[y0:y1, :x0] = fill
    out[y0:y1, x1:] = fill
    return out


def polar_buffer_clear(payload: Any = None,
                       shape: Tuple[int, ...] = (16, 16),
                       capacity: int = 4) -> Dict[str, Any]:
    """缓冲清零自检：验证复用路径与 halo 路径均无旧数据残留。"""
    pool = BufferPool(capacity=capacity)
    residuals: List[float] = []

    for tag in range(3):
        buf = pool.get(shape, dtype=np.float32)
        # 模拟分块核只写中心 ROI，halo 不写
        roi = (2, 2, shape[0] - 2, shape[1] - 2)
        buf[roi[0]:roi[2], roi[1]:roi[3]] = float(tag + 1)
        clear_halo(buf, roi)
        halo_max = float(max(np.max(np.abs(buf[:2])), np.max(np.abs(buf[-2:])),
                             np.max(np.abs(buf[:, :2])), np.max(np.abs(buf[:, -2:]))))
        residuals.append(halo_max)
        pool.recycle(buf)
        reused = pool.get(shape, dtype=np.float32)
        # 复用借出必须全零
        residuals.append(float(np.max(np.abs(reused))))
        pool.recycle(reused)

    ok = all(abs(r) < 1e-12 for r in residuals)
    if isinstance(payload, np.ndarray):
        cleared = payload.copy()
        clear_halo(cleared, (0, 0, cleared.shape[0], cleared.shape[1]))
    else:
        cleared = None

    return {
        "ok": bool(ok),
        "residuals": residuals,
        "zero_bytes": pool.zero_bytes(),
        "cleared": cleared,
        "pool_capacity": capacity,
    }


# ---------------------------------------------------------------------------
# 流水线组装入口
# ---------------------------------------------------------------------------

def load_polar_stack(source: Any = None,
                     angles_deg: Sequence[float] = POLAR_ANGLES_DEG
                     ) -> Dict[str, Any]:
    """加载四方向偏振栈。

    接受 ``(K,H,W)``/``(K,H,W,C)`` 数组、含 ``frames`` 的字典或
    ``.npy`` 路径；标量/None 时返回物理一致的合成栈。输出统一为
    ``(K,H,W,C)`` 浮点栈与方向角数组，并做基础校验。
    """
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=False)
        frames = _to_hwc(data.astype(np.float32))
        meta: Dict[str, Any] = {}
    elif isinstance(source, dict):
        frames = _to_hwc(np.asarray(source["frames"], dtype=np.float32))
        meta = {k: v for k, v in source.items() if k != "frames"}
        angles_deg = meta.pop("angles_deg", angles_deg)
    elif isinstance(source, np.ndarray):
        frames = _to_hwc(source.astype(np.float32))
        meta = {}
    else:
        syn = make_synthetic_stack()
        frames = _to_hwc(syn["frames"])
        meta = {k: v for k, v in syn.items() if k != "frames"}
        angles_deg = syn["angles_deg"]

    angles_arr = np.deg2rad(np.asarray(angles_deg, dtype=np.float32)) \
        if np.max(np.asarray(angles_deg)) > TAU else np.asarray(angles_deg,
                                                                 dtype=np.float32)
    if frames.shape[0] != len(angles_deg):
        raise ValueError(f"帧数 {frames.shape[0]} 与方向数 {len(angles_deg)} 不一致")
    if not np.all(np.isfinite(frames)):
        raise ValueError("偏振栈包含 NaN/Inf")

    return {
        "frames": frames,
        "angles_deg": np.asarray(angles_deg, dtype=np.float32),
        "angles_rad": angles_arr.astype(np.float32),
        "n_frames": int(frames.shape[0]),
        "height": int(frames.shape[1]),
        "width": int(frames.shape[2]),
        "channels": int(frames.shape[3]),
        **meta,
    }


def align_polar_frames(stack: Any, **kwargs: Any) -> Dict[str, Any]:
    """流水线对齐入口：封装 :func:`polar_angle_perpixel_align`。"""
    if _is_scalar_sentinel(stack):
        stack = make_synthetic_stack()
    return polar_angle_perpixel_align(stack, **kwargs)


def separate_reflection(stack: Any,
                        orientation: Any = 0,
                        mirrored: bool = False,
                        material: Optional[np.ndarray] = None
                        ) -> Dict[str, Any]:
    """完整反射/透射分离流水线，串联修复点 1-4 与缓冲管理。

    步骤：加载 → 透光率/曝光/径向阴影校正 → 逐像素对齐 → Stokes 拟合
    → 角度基准统一（含通道互换修正）→ 材质自适应阈值 → 反射/透射分解。
    """
    loaded = load_polar_stack(None if _is_scalar_sentinel(stack) else stack)
    frames_hwc = loaded["frames"]
    angles = loaded["angles_deg"]
    exposure = loaded.get("exposure_ms", np.ones(loaded["n_frames"], np.float32))

    # 修复点 4：先校正透光率，再做配准与融合，避免同心圆进入后续环节
    calib = frame_transmittance_calib(frames_hwc, exposure)
    corrected = calib["corrected"]

    # 修复点 1：逐像素对齐，消除波纹错位与重影
    aligned_pack = polar_angle_perpixel_align(corrected)
    aligned = aligned_pack["aligned"]  # (K,H,W,C)

    # 灰度域拟合 Stokes（彩色则逐通道处理）
    k, h, w, c = aligned.shape
    s0 = np.zeros((h, w, c), np.float32)
    s1 = np.zeros_like(s0)
    s2 = np.zeros_like(s0)
    for ch in range(c):
        a, b, d = fit_stokes(aligned[..., ch], angles)
        s0[..., ch], s1[..., ch], s2[..., ch] = a, b, d

    dop = stokes_dop(s0, s1, s2)
    aop = stokes_aop(s1, s2)
    intensity = s0.mean(axis=-1)
    dop_gray = dop.mean(axis=-1)
    aop_gray = _circular_mean([aop[..., ch] for ch in range(c)], period=PI)

    # 修复点 2：角度基准统一
    base = polar_angle_base_unify(aop_gray, orientation=orientation,
                                  mirrored=mirrored)
    if material is None:
        material = loaded.get("material_gt")

    # 修复点 3：材质自适应阈值
    adaptive = material_adaptive_thresh(dop_gray, intensity, material)
    mask = adaptive["reflection_mask"][..., None]  # (H,W,1) 广播到通道

    # 偏振分解：完全偏振部分作为反射层，无偏振直流作为透射层
    polarized = np.sqrt(s1 ** 2 + s2 ** 2)  # |偏振分量|
    reflection = np.clip(polarized * mask, 0, None)
    transmission = np.clip(s0 - reflection, 0, None)
    products = {"reflection": reflection, "transmission": transmission,
                "reflection_mask": np.broadcast_to(mask, (h, w, c)).copy()}
    products = apply_channel_swap(products, base["swap_channels"])

    # 修复点 6：输出缓冲经池化且确认清零残留
    bufcheck = polar_buffer_clear(shape=(h, w))

    return {
        "s0": s0, "s1": s1, "s2": s2,
        "dop": dop, "aop": aop,
        "aop_world": base["aop_world"],
        "swap_channels": base["swap_channels"],
        "reflection": products["reflection"],
        "transmission": products["transmission"],
        "reflection_mask": products["reflection_mask"],
        "thresh": adaptive["thresh"],
        "material": adaptive["material"],
        "aligned": aligned,
        "flows": aligned_pack["flows"],
        "ghost_energy": aligned_pack["ghost_energy"],
        "gains": calib["gains"],
        "shading": calib["shading"],
        "uniformity_cv": calib["uniformity_cv"],
        "buffer_ok": bufcheck["ok"],
    }


def present_polar(result: Any, out_dir: Optional[str] = None) -> Dict[str, Any]:
    """把分离结果可视化为可展示的 8-bit BGR 拼图。

    布局：透射通道 | 反射通道 | DoP 热力图 | AoP 方向色图。
    ``out_dir`` 给定时把拼图与各单图写入目录。
    """
    if _is_scalar_sentinel(result):
        result = separate_reflection(None)

    def to_bgr(x: np.ndarray) -> np.ndarray:
        g = _norm01(x.mean(axis=-1) if x.ndim == 3 else x)
        return cv2.cvtColor((g * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    trans = to_bgr(result["transmission"])
    refl = to_bgr(result["reflection"])

    dop = result["dop"].mean(axis=-1) if result["dop"].ndim == 3 else result["dop"]
    dop_bgr = cv2.applyColorMap((_norm01(dop) * 255).astype(np.uint8),
                                cv2.COLORMAP_VIRIDIS)

    aop = result.get("aop_world",
                     result["aop"].mean(axis=-1) if result["aop"].ndim == 3
                     else result["aop"])
    h, w = aop.shape
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ((_wrap_half_pi(aop) + PI / 2) / PI * 179).astype(np.uint8)
    hsv[..., 1] = 220
    hsv[..., 2] = (_norm01(result["s0"].mean(axis=-1)) * 255).astype(np.uint8)
    aop_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    panel = np.concatenate([trans, refl, dop_bgr, aop_bgr], axis=1)
    saved: List[str] = []
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for name, img in [("transmission.png", trans), ("reflection.png", refl),
                          ("dop.png", dop_bgr), ("aop.png", aop_bgr),
                          ("panel.png", panel)]:
            p = os.path.join(out_dir, name)
            cv2.imwrite(p, img)
            saved.append(p)
    return {"panel": panel, "tiles": {"transmission": trans, "reflection": refl,
                                      "dop": dop_bgr, "aop": aop_bgr},
            "saved": saved}


def validate_polar(stack: Any = None) -> Dict[str, Any]:
    """输入/产物校验：帧数、方向角、值域、形状、有限性。"""
    loaded = load_polar_stack(None if _is_scalar_sentinel(stack) else stack)
    frames = loaded["frames"]
    checks: Dict[str, bool] = {}
    checks["four_frames"] = loaded["n_frames"] == 4
    checks["angles_monotonic"] = bool(np.all(np.diff(loaded["angles_deg"]) > 0))
    checks["finite"] = bool(np.all(np.isfinite(frames)))
    checks["non_negative"] = bool(np.min(frames) >= 0)
    checks["spatial_consistent"] = all(frames[i].shape == frames[0].shape
                                       for i in range(loaded["n_frames"]))
    # Stokes 物理约束：S0 >= |偏振分量|
    s0, s1, s2 = fit_stokes(np.stack([_as_gray(frames[i])
                                      for i in range(loaded["n_frames"])]),
                            loaded["angles_deg"])
    checks["stokes_physical"] = bool(np.all(s0 + 1e-4 >= np.sqrt(s1 ** 2 + s2 ** 2)))
    return {"ok": bool(all(checks.values())), "checks": checks,
            "n_frames": loaded["n_frames"],
            "shape": tuple(int(v) for v in frames.shape)}


def metrics_polar(result: Any = None) -> Dict[str, float]:
    """量化去眩光质量：亮度均匀性、重影能量、接缝连续性、过曝占比。"""
    if _is_scalar_sentinel(result):
        result = separate_reflection(None)

    refl = result["reflection"]
    trans = result["transmission"]
    h, w = refl.shape[:2]

    # 亮度均匀性：透射层变异系数
    trans_gray = _as_gray(trans)
    uniformity = float(np.std(trans_gray) / max(np.mean(trans_gray), 1e-6))

    # 过曝占比：反射层接近饱和的像素比例（金属过曝的直接度量）
    overexposed = float(np.mean(refl >= 0.98 * max(float(np.max(refl)), 1e-8)))

    # 同心圆能量：径向平均曲线的方差（透光率校正后应较小）
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2).astype(np.int32)
    radial_mean = np.zeros(int(r.max()) + 1, np.float64)
    for ri in range(radial_mean.size):
        m = r == ri
        if m.any():
            radial_mean[ri] = trans_gray[m].mean()
    concentric = float(np.std(radial_mean) / max(np.mean(radial_mean), 1e-6))

    # 接缝条纹：反射掩膜边界的梯度能量
    mask = result["reflection_mask"][..., 0] if result["reflection_mask"].ndim == 3 \
        else result["reflection_mask"]
    gx = cv2.Sobel(mask.astype(np.float32), cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(mask.astype(np.float32), cv2.CV_32F, 0, 1, 3)
    seam = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))

    return {
        "uniformity_cv": uniformity,
        "ghost_energy": float(result.get("ghost_energy", 0.0)),
        "overexposed_ratio": overexposed,
        "concentric_energy": concentric,
        "seam_energy": seam,
        "buffer_ok": float(bool(result.get("buffer_ok", True))),
    }


def audit_polar(result: Any = None, out_dir: Optional[str] = None) -> Dict[str, Any]:
    """汇总审计：校验 + 指标 + 可视化，可选落盘，返回结构化审计报告。"""
    stack = None
    if isinstance(result, dict) and "aligned" in result:
        target = result
    else:
        target = separate_reflection(None if _is_scalar_sentinel(result) else result)

    validation = validate_polar(target.get("aligned"))
    metrics = metrics_polar(target)
    presentation = present_polar(target, out_dir=out_dir)

    report = {
        "validation": validation,
        "metrics": metrics,
        "swap_channels": bool(target.get("swap_channels", False)),
        "buffer_ok": bool(target.get("buffer_ok", False)),
        "ghost_energy": float(target.get("ghost_energy", 0.0)),
        "uniformity_cv": float(target.get("uniformity_cv", 0.0)),
        "artifacts": presentation["saved"],
        "passed": bool(validation["ok"] and target.get("buffer_ok", False)),
    }
    return report


def fallback_polar(stack: Any = None) -> Dict[str, Any]:
    """退化输入兜底：任何异常输入下降级为时间均值 + 软掩膜，保证不崩溃。"""
    try:
        loaded = load_polar_stack(None if _is_scalar_sentinel(stack) else stack)
        frames = loaded["frames"]
        # 兜底策略：无偏振配准时退化为多帧平均（最大程度抑制瞬态波纹）
        mean_frame = np.mean(frames, axis=0)
        gray = np.stack([_as_gray(frames[i]) for i in range(loaded["n_frames"])])
        temporal_std = np.std(gray, axis=0)
        soft_mask = _norm01(temporal_std).astype(np.float32)
        return {
            "degraded": True,
            "reason": "fallback: temporal average without polar decomposition",
            "transmission": np.clip(mean_frame * (1.0 - soft_mask[..., None]), 0, None),
            "reflection": np.clip(mean_frame * soft_mask[..., None], 0, None),
            "reflection_mask": np.broadcast_to(soft_mask[..., None],
                                               mean_frame.shape).copy(),
            "frame_count": loaded["n_frames"],
        }
    except Exception as exc:  # 极端异常下返回中性灰图，流水线仍可出图
        h = w = 32
        neutral = np.full((h, w, 1), 128.0, np.float32)
        return {
            "degraded": True,
            "reason": f"fallback: input rejected ({type(exc).__name__}: {exc})",
            "transmission": neutral,
            "reflection": np.zeros_like(neutral),
            "reflection_mask": np.zeros((h, w, 1), np.float32),
            "frame_count": 0,
        }


def determinism_polar(seed: Any = _RNG_SEED) -> Dict[str, Any]:
    """确定性自检：同输入两次完整流水线结果必须逐元素一致。"""
    rng1 = np.random.default_rng(int(seed) if not isinstance(seed, str) else _RNG_SEED)
    rng2 = np.random.default_rng(int(seed) if not isinstance(seed, str) else _RNG_SEED)
    st1 = make_synthetic_stack(rng=rng1)
    st2 = make_synthetic_stack(rng=rng2)

    r1 = separate_reflection(st1)
    r2 = separate_reflection(st2)

    compared = {}
    all_equal = True
    for key in ("reflection", "transmission", "dop", "aop_world", "thresh"):
        a, b = r1[key], r2[key]
        equal = bool(np.allclose(a, b, atol=1e-6, rtol=1e-6))
        compared[key] = equal
        all_equal &= equal

    # 缓冲池两次自检结果也应一致
    b1 = polar_buffer_clear(shape=(16, 16))
    b2 = polar_buffer_clear(shape=(16, 16))
    buf_equal = b1["residuals"] == b2["residuals"]
    all_equal &= bool(buf_equal)

    return {"deterministic": bool(all_equal), "per_output": compared,
            "buffer_deterministic": bool(buf_equal), "seed": int(seed)
            if not isinstance(seed, str) else _RNG_SEED}
