# AI Piano Evaluation

**钢琴动画评测方法与独立工具 · MIT · v0.1.0**

评估 MIDI 驱动的钢琴演奏动画，而不是音频转 MIDI。提供协议、指标实现、输入规范、
合成反例和测试；**不提供模型、训练/推理代码、真实曲库、动作录制、角色资产或逆向资料**。
这是全新独立实现，不是把私有项目历史或代码目录改为公开。

> **公开的是“怎么测”，不是“模型已经多好”。**
> 合成样例只验证计算是否正确。没有公开真实模型成绩，不把程序通过等同于动画自然。

## 测什么

| 维度 | 已实现 | 不能据此宣称 |
|---|---|---|
| 参考动作误差 | 坐标分量 RMSE、平均三维点距离、旋转角误差 | 唯一正确/自然的演奏姿态 |
| 指法参考一致性 | 10 类指法与左右手准确率、混淆矩阵、弃权覆盖率 | 与参考不同就一定不能演奏 |
| 时序 | 按相同事件 ID 的 onset 偏差（毫秒） | 音频听感或完整释键/踏板评测 |
| 参考指法条件下触键 | 指尖点落在目标键 XY 区域、与键顶 Z 的接近程度 | 全琴键覆盖率或完整网格无穿模 |
| 运动变化 | 速度、加速度、jerk，按秒计；连续持键片段单独统计 | 越静止越自然 |
| 汇总与分组 | 按计数汇总的 micro、整段演奏等权的 macro、分组泄漏声明检查 | 已经证明数据绝无泄漏 |

**尚未实现**：真实皮肤/琴键/手指自碰撞检测、音符级多指替代触键匹配、完整释键/踏板指标、
自动判断演奏自然度。它们明确显示 `not_measured`，不是 0，也不是通过。

## 快速开始

Python 3.9+，仅需 NumPy：

```sh
python -m venv .venv
# 激活虚拟环境后
python -m pip install -e .
python examples/generate_synthetic.py .demo
python -m piano_eval evaluate \
  --prediction .demo/identical.npz --reference .demo/reference.npz \
  --metadata .demo/metadata.json --output .demo/identical-report.json
python -m piano_eval evaluate \
  --prediction .demo/floating.npz --reference .demo/reference.npz \
  --metadata .demo/metadata.json --output .demo/floating-report.json
python -m unittest discover -s tests -v
```

示例目录须不存在，报告拒绝覆盖。`identical` 仅表示与人工构造参考相同；它不是 AiPiano 的真实输出。
另外生成 `jitter.npz` 和 `wrong_hand.npz` 两个反例。没有网络、账号、模型下载或远程计费。

## 必须先固定的评测条件

1. 按曲目组、录制演奏分组切分；变速、移调、同曲片段/改编不能跨训练与测试。
2. 冻结测试清单、模型版本、后处理参数、采样与阈值，再评测；调参看过的集合只能算开发集。
3. 单位为米，坐标为键盘右手坐标系；预测和参考时间戳、点 ID、事件 ID 必须一致。
   工具**不自动平移时间、缩放骨架、对齐根节点或做 Procrustes 拟合**。
4. 原始预测和最终 IK/约束/重定向后的动画分开；使用参考指法的 teacher-forced 结果和端到端结果分开。
5. 测试最终动画时，输入应来自宿主求值后的世界空间几何/骨骼，不只是 IK 控制器目标。
6. 缺少的测量报告缺失；冻结动作可以拥有很低 jerk，必须同时看接触、姿态与盲评。

## 文档

- [完整评测协议](docs/PROTOCOL.md)：实验设计、负例、人工盲评与结果发布
- [指标公式与边界](docs/METRICS.md)
- [输入文件规范](docs/DATA_SPEC.md)
- [Blender / C4D 导出接入清单](docs/ADAPTER_CHECKLIST.md)（指导，不含已验证宿主插件）
- [公开范围与来源声明](docs/PUBLICATION_SCOPE.md)
- [本版本验证记录](docs/VALIDATION.md)

本协议把位置误差与动作质量分开。MPJPE 的一般定义可参考
[Human3.6M 官方协议](https://vision.imar.ro/human3.6m/readme_challenge.php)；
本项目不沿用其人体骨架、根节点对齐或数据，因此结果不能直接与该排行榜比较。
钢琴动作的多维评估背景可参考 [PianoMotion10M 论文](https://arxiv.org/abs/2406.09326)。
这些链接不表示获得数据再分发许可，也不表示实现了论文中的全部指标。

## English summary

A model-independent, offline protocol and NumPy toolkit for evaluating piano-performance animation.
It distinguishes reference pose agreement, fingering, event timing, a clearly labeled point-to-key-box contact proxy,
and time-normalized motion derivatives. It reports missing measurements explicitly and provides no universal
quality score. Only synthetic examples are distributed. Models, recorded performances and proprietary assets
are not included. See the data contract and protocol before adapting your outputs.

[MIT License](LICENSE). No affiliation with third-party datasets, DCC vendors or their authors is implied.
