# pairwise-polarization-deglare

## 领域

游戏、图形与多媒体

## 主题

偏振图像去眩光与反射分离

## 问题与修复

四方向（0°/45°/90°/135°）偏振片顺序采集，六个失败模式与对应实现：

| # | 失败现象 | 根因 | 修复入口与方法 |
|---|---------|------|----------------|
| 1 | 去眩光后重影 | 单应矩阵只表达全局平面变换，水面波纹的局部非刚性位移无法对齐 | `polar_angle_perpixel_align`：单应矩阵给全局基准，Farneback 稠密光流估计逐像素位移场，反向重映射对齐 |
| 2 | 横竖屏切换后反射/透射通道互换、天空被错误增强 | 偏振角按相机旋转读取，角度基准翻转 | `polar_angle_base_unify`：按设备方向/镜像把 AoP 旋回世界基准（π 周期绕回），输出 `swap_channels` 标志并由 `apply_channel_swap` 交换通道 |
| 3 | 金属过曝、水面反射残留 | 金属低 DoP、水面高 DoP，全局单一阈值不可兼得 | `material_adaptive_thresh`：材质基准阈值 + 局部 DoP/亮度自适应，逐像素阈值图与软掩膜 |
| 4 | 融合亮度不均、边缘同心圆 | 偏振片旋转使逐帧透光率不同，按曝光时间加权失效 | `frame_transmittance_calib`：曝光归一化 → 近零 DoP 锚点（无偏振参考带）估计逐帧增益，两轮精化 → 径向阴影场移除同心圆 |
| 5 | 拼接缝条纹、块边界反射强度跳变 | 重叠区各自估计偏振参数，互不约束 | `polar_overlap_joint_solve`：距离羽化加权联合求解，AoP 走圆周平均（尊重 π 周期） |
| 6 | 缓冲复用残留旧数据 | GPU 工作缓冲复用未清零 | `polar_buffer_clear` + `BufferPool`：借出零填充、归还即清零，halo 区显式清零并自检 |

## 流水线

`load_polar_stack → frame_transmittance_calib → polar_angle_perpixel_align
→ fit_stokes → polar_angle_base_unify → material_adaptive_thresh
→ 反射/透射分解 → present_polar / metrics_polar / audit_polar`

辅助入口：`validate_polar`（物理约束校验）、`fallback_polar`（退化输入兜底）、
`determinism_polar`（同输入逐元素确定性自检）。

## 测试

```bash
python -m pytest -q
```

- `tests/test_polarization.py`：15 个公开入口的契约测试；
- `tests/test_pipeline_physics.py`：基于物理一致合成偏振栈的 19 项验证，
  覆盖六个修复点的失败模式（重影能量、通道互换、材质阈值分化、
  透光率增益相关性、接缝连续性、缓冲零残留）与端到端不变量。

全部通过（34 passed）。
