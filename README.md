# pairwise-polarization-deglare

## 领域

游戏、图形与多媒体

## 主题

偏振图像去眩光与反射分离

## 说明

逐像素偏振角对齐、角度基准统一、材质自适应阈值。

初始状态下 `imaging/polarization.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
