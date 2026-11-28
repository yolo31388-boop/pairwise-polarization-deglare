"""六个修复点与端到端流水线的物理验证（基于物理一致的合成偏振栈）。"""

import math

import cv2
import numpy as np
import pytest

from imaging import polarization as mod


@pytest.fixture(scope="module")
def syn():
    return mod.make_synthetic_stack(h=48, w=48, ripple=True,
                                    rng=np.random.default_rng(7))


# --- 修复点 1：逐像素偏振角对齐 -------------------------------------------

def test_perpixel_align_beats_homography(syn):
    """波纹为非刚性位移：稠密光流对齐后的重影能量必须低于纯单应对齐。"""
    dense = mod.polar_angle_perpixel_align(syn, use_homography=True)
    homo = mod.polar_angle_perpixel_align(syn, use_homography=False)
    assert dense["aligned"].shape == syn["frames"].shape + (1,)
    assert dense["flows"].shape == (4, 48, 48, 2)
    # 光流场在波纹区非零，说明确实做了逐像素（而非全局）位移估计
    assert np.max(np.linalg.norm(dense["flows"], axis=-1)) > 0.1
    assert dense["ghost_energy"] <= homo["ghost_energy"]


def test_align_reference_frame_untouched(syn):
    out = mod.polar_angle_perpixel_align(syn, ref_index=0)
    # 参考帧位移场为零
    assert np.allclose(out["flows"][0], 0.0, atol=1e-6)


# --- 修复点 2：角度基准统一 -----------------------------------------------

def test_angle_base_orientation_rotation():
    aop = np.full((8, 8), math.radians(30.0), np.float32)
    portrait = mod.polar_angle_base_unify(aop, orientation=0)
    landscape = mod.polar_angle_base_unify(aop, orientation=90)
    # 横屏相对竖屏旋转 90°，世界系 AoP 应整体减去 90°（pi/2 周期下等价翻转）
    expect = mod._wrap_half_pi(aop - math.pi / 2)
    assert np.allclose(landscape["aop_world"], expect, atol=1e-5)
    assert np.allclose(portrait["aop_world"], aop, atol=1e-5)
    assert landscape["swap_channels"] is True
    assert portrait["swap_channels"] is False


def test_angle_base_mirror_and_swap(syn):
    s0, s1, s2 = mod.fit_stokes(syn["frames"], syn["angles_deg"])
    aop = mod.stokes_aop(s1, s2)
    mirrored = mod.polar_angle_base_unify(aop, orientation=0, mirrored=True)
    assert mirrored["swap_channels"] is True
    assert np.allclose(mirrored["aop_world"], mod._wrap_half_pi(-aop), atol=1e-5)
    # 180° 旋转不是 90° 奇数倍，不触发通道互换
    rot180 = mod.polar_angle_base_unify(aop, orientation=180)
    assert rot180["swap_channels"] is False


def test_channel_swap_exchanges_layers():
    refl = np.ones((4, 4, 1), np.float32)
    trans = np.zeros((4, 4, 1), np.float32)
    mask = np.ones((4, 4, 1), np.float32)
    out = mod.apply_channel_swap(
        {"reflection": refl, "transmission": trans, "reflection_mask": mask}, True)
    assert np.allclose(out["reflection"], 0.0)
    assert np.allclose(out["transmission"], 1.0)
    assert np.allclose(out["reflection_mask"], 0.0)


# --- 修复点 3：材质自适应阈值 ---------------------------------------------

def test_material_thresholds_differ(syn):
    s0, s1, s2 = mod.fit_stokes(syn["frames"], syn["angles_deg"])
    dop = mod.stokes_dop(s0, s1, s2)
    out = mod.material_adaptive_thresh(dop, s0, syn["material_gt"])
    metal = out["thresh"][syn["material_gt"] == 0]
    water = out["thresh"][syn["material_gt"] == 1]
    # 金属阈值显著低于水面阈值，避免“金属过曝、水面残留”
    assert float(np.mean(metal)) < float(np.mean(water)) - 0.05
    assert out["thresh"].shape == dop.shape
    assert set(np.unique(out["reflection_mask"])).issubset({0.0, 1.0})


def test_material_threshold_without_prior(syn):
    s0, s1, s2 = mod.fit_stokes(syn["frames"], syn["angles_deg"])
    dop = mod.stokes_dop(s0, s1, s2)
    out = mod.material_adaptive_thresh(dop, s0, material=None)
    assert set(np.unique(out["material"])).issubset({0, 1, 2})
    assert np.all((out["thresh"] >= 0.03) & (out["thresh"] <= 0.60))


# --- 修复点 4：逐帧透光率校正 ---------------------------------------------

def test_transmittance_calib_flattens_frames():
    # 显式注入透光率差异、径向阴影与噪声的栈
    syn = mod.make_synthetic_stack(h=48, w=48, ripple=False, transmittance=True,
                                   shading=True, rng=np.random.default_rng(3))
    out = mod.frame_transmittance_calib(syn["frames"], syn["exposure_ms"])
    assert out["corrected"].shape == syn["frames"].shape  # 灰度输入保持三维
    # 校正后各帧中位数一致：变异系数远小于原始帧间差异
    raw_med = [np.median(syn["frames"][i]) for i in range(4)]
    raw_cv = float(np.std(raw_med) / np.mean(raw_med))
    assert out["uniformity_cv"] < raw_cv * 0.5
    # 估计增益与合成时注入的透光率倒数趋势一致（相关即可，容差宽松）
    assert np.corrcoef(out["gains"], 1.0 / syn["trans_gt"])[0, 1] > 0.9


def test_transmittance_exposure_normalization():
    frames = np.repeat(np.full((1, 16, 16), 100.0, np.float32), 4, axis=0)
    exposures = np.array([10.0, 20.0, 10.0, 20.0], np.float32)
    out = mod.frame_transmittance_calib(frames, exposures, correct_shading=False)
    # 曝光归一化后 20ms 帧亮度减半，再被增益拉平
    post = [np.median(out["corrected"][i]) for i in range(4)]
    assert np.std(post) / np.mean(post) < 0.05


# --- 修复点 5：重叠区联合求解 ---------------------------------------------

def test_overlap_joint_solve_continuity():
    h = w = 40
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dop = 0.3 + 0.002 * xx
    aop = np.full((h, w), 0.4, np.float32)
    tiles = [
        {"y0": 0, "x0": 0, "y1": h, "x1": 28,
         "dop": dop[:, :28] + 0.05, "aop": aop[:, :28] + 0.1},
        {"y0": 0, "x0": 12, "y1": h, "x1": w,
         "dop": dop[:, 12:] - 0.05, "aop": aop[:, 12:] - 0.1},
    ]
    out = mod.polar_overlap_joint_solve(tiles, canvas_shape=(h, w))
    merged = out["merged"]
    assert merged["dop"].shape == (h, w)
    # 重叠区取值落在两块估计之间（联合求解，而非取某一边）
    overlap_col = 20
    v = merged["dop"][:, overlap_col]
    assert np.all(v > dop[:, overlap_col] - 0.05 - 1e-4)
    assert np.all(v < dop[:, overlap_col] + 0.05 + 1e-4)
    # 非重叠区保持块原值
    assert np.allclose(merged["dop"][:, 0], dop[:, 0] + 0.05, atol=1e-5)
    assert out["overlap_mask"].sum() > 0
    assert "dop" in out["seam_scores"]


def test_overlap_circular_aop_average():
    """AoP 接近 ±pi/2 边界时圆周平均不能跨周期产生错误跳变。"""
    tiles = [
        {"y0": 0, "x0": 0, "y1": 8, "x1": 6,
         "aop": np.full((8, 6), 1.55, np.float32)},
        {"y0": 0, "x0": 2, "y1": 8, "x1": 8,
         "aop": np.full((8, 6), -1.55, np.float32)},
    ]
    out = mod.polar_overlap_joint_solve(tiles, canvas_shape=(8, 8))
    aop = out["merged"]["aop"]
    # 1.55 与 -1.55 在 pi 周期下几乎同向，融合结果也应贴近 ±pi/2，而非 0
    assert np.all(np.abs(aop[:, 2:6]) > 1.4)


# --- 修复点 6：缓冲清零 ---------------------------------------------------

def test_buffer_pool_reuse_is_zeroed():
    pool = mod.BufferPool(capacity=2)
    buf = pool.get((16, 16))
    buf[:] = 7.0
    pool.recycle(buf)
    reused = pool.get((16, 16))
    assert reused is buf
    assert np.all(reused == 0.0)  # 复用借出已清零，无旧数据残留


def test_buffer_clear_halo_and_selfcheck():
    buf = np.ones((12, 12), np.float32)
    mod.clear_halo(buf, (2, 2, 10, 10))
    assert np.all(buf[:2] == 0.0) and np.all(buf[-2:] == 0.0)
    assert np.all(buf[:, :2] == 0.0) and np.all(buf[:, -2:] == 0.0)
    assert np.all(buf[2:10, 2:10] == 1.0)
    report = mod.polar_buffer_clear(shape=(12, 12))
    assert report["ok"] is True
    assert all(abs(r) < 1e-12 for r in report["residuals"])


# --- 端到端流水线 ---------------------------------------------------------

def test_separate_reflection_end_to_end(syn):
    result = mod.separate_reflection(syn, orientation=90)
    h, w = syn["frames"].shape[1:]
    assert result["reflection"].shape[:2] == (h, w)
    assert result["transmission"].shape[:2] == (h, w)
    # 能量守恒：反射 + 透射不超过总强度 S0
    assert np.all(result["reflection"] + result["transmission"]
                  <= result["s0"] + 1e-3)
    assert result["swap_channels"] is True
    assert result["buffer_ok"] is True
    # DoP 物理范围
    assert np.all((result["dop"] >= 0) & (result["dop"] <= 1.0 + 1e-6))


def test_validate_and_metrics(syn):
    result = mod.separate_reflection(syn)
    report = mod.validate_polar(syn)
    assert report["ok"] is True
    metrics = mod.metrics_polar(result)
    for key in ("uniformity_cv", "ghost_energy", "overexposed_ratio",
                "concentric_energy", "seam_energy"):
        assert math.isfinite(metrics[key])
        assert metrics[key] >= 0.0


def test_present_polar_panel(syn):
    result = mod.separate_reflection(syn)
    vis = mod.present_polar(result)
    # 四联拼图宽度为单图 4 倍，高度一致
    assert vis["panel"].shape[0] == result["reflection"].shape[0]
    assert vis["panel"].shape[1] == 4 * result["reflection"].shape[1]


def test_fallback_handles_bad_input():
    out = mod.fallback_polar(np.zeros((1, 4, 4), np.float32))  # 帧数与方向数不符
    assert out["degraded"] is True
    assert "transmission" in out
    # 极端异常也不抛出
    out2 = mod.fallback_polar({"frames": "not-an-array"})
    assert out2["degraded"] is True


def test_determinism():
    report = mod.determinism_polar(seed=42)
    assert report["deterministic"] is True
    assert all(report["per_output"].values())


def test_stokes_recovers_synthetic_params():
    """无退化干净栈上，Stokes 拟合应恢复注入的 AoP/DoP 真值。"""
    clean = mod.make_synthetic_stack(h=48, w=48, ripple=False,
                                     transmittance=False, shading=False,
                                     noise_sigma=0.0, rng=np.random.default_rng(1))
    s0, s1, s2 = mod.fit_stokes(clean["frames"], clean["angles_deg"])
    aop = mod.stokes_aop(s1, s2)
    dop = mod.stokes_dop(s0, s1, s2)
    # 无偏振参考带（DoP=0）上 AoP 无定义，只校验有偏振的区域
    polarized = clean["dop_gt"] > 0.01
    assert np.allclose(aop[polarized], clean["aop_gt"][polarized], atol=0.02)
    assert np.allclose(dop, clean["dop_gt"], atol=0.02)
    assert np.allclose(s0, clean["s0_gt"], rtol=0.03)
