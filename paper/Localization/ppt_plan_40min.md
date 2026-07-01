# 40分钟组会PPT计划

目标：做一套 **16页主讲 + 2-3页备份** 的 PPT，适合“混合背景、方法为主、只放静态实验结果”的汇报场景。

总节奏建议：**背景 10 分钟 + 方法 18 分钟 + 实验 8 分钟 + 总结 4 分钟**

## 主讲页

### 1. 封面页（0.5 分钟）
- 标题、作者、组会日期、项目名。
- 图：论文主图或方法框架小图。

### 2. 汇报大纲（0.5 分钟）
- 背景
- 方法
- 实验
- 总结

### 3. VPR 与分层定位背景（3.5 分钟）
- 讲清 VPR 在 Hierarchical Localization 中的位置。
- 全局检索如何影响后续匹配和位姿估计。
- 为什么“检索质量”是上限。
- 图：`framework` 总体流程图。

### 4. 现有方法痛点（3.5 分钟）
- NetVLAD 类方法的一阶聚合局限。
- 局部特征提取与全局检索的矛盾。
- 为什么需要分布建模和质量约束。
- 图：传统聚合 vs OT 聚合示意。

### 5. 论文贡献概览（1.5 分钟）
- OT 分配。
- 高斯残差描述子。
- Wasserstein 几何解释。
- OT-anchored refinement。

### 6. 整体方法框架图（2 分钟）
- backbone -> OT assignment -> moments -> residual descriptor -> refinement
- 用一张图串起全链路。
- 图：`ot_gaussvlad2_arch.pdf` 或自绘总框图。

### 7. A：OT 运输分配原理（4 分钟）
- score matrix、dustbin、Sinkhorn。
- 重点强调“全局质量约束”不是 token-wise softmax。
- 图：运输矩阵和 dustbin 位置。

### 8. A 的数学表达（1.5 分钟）
- \(A^\star = \mathrm{diag}(u)K\mathrm{diag}(v)\)
- 只讲直觉，不深挖推导。

### 9. B：Transport-induced Gaussian moments（3 分钟）
- \(\hat{\mu}_k\)、\(\hat{v}_k\)、\(\hat{\sigma}_k\) 的含义。
- 解释它们代表“当前图像里这个簇的分布统计”。
- 图：cluster mass -> moment estimation。

### 10. C：Wasserstein residual descriptor（3 分钟）
- 残差如何构造。
- 为什么对角高斯下等价于欧氏嵌入。
- 为什么这比普通 VLAD 更有表达力。
- 图：mean / std residual 拼接。

### 11. D：OT-anchored refinement 动机（1.5 分钟）
- 为什么 refinement 不能直接替代 OT。
- 为何要“先 OT，再修正”。

### 12. D：refiner 结构细节（4 分钟）
- `LayerNorm -> Linear -> GELU -> Dropout -> Linear`
- `MultiheadAttention`
- 可选 FFN
- 零初始化 head
- `inverse_sigmoid` 小门控
- `max_residual_scale` 约束
- 图：refiner 小模块图。

### 13. 训练与实现设置（2 分钟）
- 数据集。
- backbone。
- cluster 数、维度、Sinkhorn 步数。
- `mass_preserving` 等关键设置。
- 图：参数表。

### 14. 实验主结果（4 分钟）
- 只放静态表格/曲线。
- 强调收敛速度、Recall、定位效果。
- 图：训练损失曲线 + recall / localization 表。

### 15. 消融分析（3 分钟）
- OT / Gaussian query / refinement / mass preserving 的贡献。
- 解释每个模块带来的增益。
- 图：ablation 表格。

### 16. 总结与展望（2 分钟）
- 一句话总结贡献。
- 后续可扩展方向。
- 图：一句话结论 + take-home message。

## 备份页

### A1. 关键公式页
- OT 目标函数。
- Gaussian moments。
- Wasserstein residual。

### A2. 参数页
- 数据集。
- 超参数。
- 训练时长。
- 实现细节。

### A3. 额外消融页
- 更多 ablation 图。
- 不同超参数的稳定性。

## 讲述策略

- 听众是混合背景，不默认大家熟悉 VPR 或 Hloc。
- 方法页优先讲直觉，再给最关键公式。
- 代码只和方法对应，不展开实现细节。
- 实验只放静态图，不放视频或动图。

## 时间检查

- 总时长控制在 38-40 分钟。
- 每页讲解时长不超过 4 分钟。
- 现场优先保证“背景讲清楚、方法讲明白、结果讲得出结论”。

