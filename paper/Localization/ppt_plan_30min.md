# 30页组会PPT重设计方案

目标：把原来的 16 页扩展成 **30 页**，仍然控制在 **40 分钟** 内，但更适合“混合背景”听众。新的版本把背景、最新 VPR 趋势、数据集、训练设置拆开讲，避免方法页过密，也方便讲清楚“VPR 在 SLAM / Hloc 中的位置”和“为什么这个方法值得做”。

总节奏建议：**背景 8 分钟 + 相关工作 5 分钟 + 方法 17 分钟 + 数据/实验 7 分钟 + 总结 3 分钟**

## 主讲页

### 1. 封面页，0.5 分钟
- 题目、作者、日期、项目名。
- 图：论文主图或总框架缩略图。

### 2. 汇报大纲，0.5 分钟
- 背景
- 相关工作
- 方法
- 数据与实验
- 总结

### 3. 什么是 VPR，1 分钟
- 视觉地点识别的任务定义。
- 输入、输出、典型应用。
- 图：query -> retrieve -> localize 的简图。

### 4. VPR 在 Hloc / SLAM 中的位置，1.5 分钟
- VPR 作为检索前端，影响后续匹配和重定位。
- 在 SLAM 里常见于回环检测、重定位、地点候选生成。
- 图：SLAM / Hloc 系统位置图。

### 5. 为什么“检索质量”是上限，1.5 分钟
- 检索错了，后面的局部匹配和 PnP 很难补回来。
- Hloc 里 coarse-to-fine 的上限由 retrieval 决定。
- 图：retrieval quality -> localization success 的链路图。

### 6. 背景小结，1 分钟
- 统一问题：全局检索要快，还要稳。
- 为后面方法动机做转场。

### 7. VPR 方法演进图，1 分钟
- 从 NetVLAD 到 BOQ / SALAD / QAA / SELA 等趋势。
- 先给“方法谱系”，不展开公式。
- 图：方法族谱或时间线。
- 下方请用小号字体附上论文

### 8. NetVLAD 的局限，1 分钟
- 一阶聚合只能表达“加权和”。
- 对局部结构、分布和不确定性刻画不足。
- 图：VLAD 聚合示意。

### 9. 最近 VPR 趋势：BOQ / SALAD / QAA / SELAVPR / SELAVPR++，1.5 分钟
- 强调最新工作在“token 选择、质量建模、聚合方式”上的方向。
- 只讲趋势和启发，不做论文综述。
- 图：对比表。

### 10. 现有方法痛点总结，1 分钟
- 一阶聚合不够。
- token 质量和全局约束没被显式建模。
- 这就是本文切入点。

### 11. 论文贡献概览，1.5 分钟
- OT 分配。
- 高斯残差描述子。
- Wasserstein 几何解释。
- OT-anchored refinement。

### 12. 整体方法框架图，1.5 分钟
- backbone -> OT assignment -> moments -> residual descriptor -> refinement。
- 图：`ot_gaussvlad2_arch.pdf` 或重绘总图。

### 13. Backbone 到 token，1 分钟
- 输入特征图如何被映射成局部 tokens。
- 为什么需要聚合前的维度压缩。
- 图：backbone + adapter。

### 14. A：OT 运输分配直觉，1.5 分钟
- score matrix、dustbin、全局质量约束。
- 核心点：不是 token-wise softmax。
- 图：运输矩阵示意。

### 15. A：Sinkhorn 与 dustbin，1.5 分钟
- 行列边际约束。
- dustbin 吸收低质量 token。
- 图：mass flow 图。

### 16. A：数学表达，0.5 分钟
- \(A^\star = \mathrm{diag}(u)K\mathrm{diag}(v)\)
- 只保留关键公式和直觉。

### 17. A：transport mass 与 token saliency，1 分钟
- 讲 token mass 的作用。
- 说明为什么需要质量分配而不是均匀平均。

### 18. B：Transport-induced Gaussian moments，1.5 分钟
- \(\hat{\mu}_k\)、\(\hat{v}_k\)、\(\hat{\sigma}_k\) 的定义。
- 它们是“当前图像里这个簇的统计量”。
- 图：cluster mass -> moment estimation。

### 19. B：moment 的直觉解释，1 分钟
- 均值是“中心在哪里”。
- 方差/标准差是“这个簇有多散”。
- 图：mean / variance 可视化。

### 20. C：Wasserstein residual descriptor，1.5 分钟
- 残差怎么从 \(\hat{\mu}_k,\hat{\sigma}_k\) 和 prototype 生成。
- 为什么比普通 VLAD 更强。
- 图：mean/std residual 拼接图。

### 21. C：为什么和 diagonal-Gaussian 几何一致，1 分钟
- 对角高斯下，Wasserstein 距离可化成欧氏形式。
- 这让 descriptor 既有几何解释又便于学习。

### 22. D：OT-anchored refinement 动机，1 分钟
- refinement 不是替代 OT，而是修正 OT 统计。
- 为什么“先 OT，再修正”更稳。

### 23. D：refiner 结构图，1.5 分钟
- `LayerNorm -> Linear -> GELU -> Dropout -> Linear`
- `MultiheadAttention`
- 可选 FFN。
- 图：refiner 模块图。

### 24. D：门控与初始化细节，1 分钟
- 零初始化 head。
- `inverse_sigmoid` 小门控。
- `max_residual_scale` 约束。
- 讲清“从接近基线开始学”。

### 25. 代码和论文的对应关系，0.5 分钟
- 只讲高层对应，不展开实现细节。
- 把 `OT`, `moments`, `refinement` 三块和代码模块对上。

### 26. 数据集总览：为什么要单独讲，1 分钟
- VPR 数据集类型很多，标注和划分不统一。
- 需要标准化 downloader 和统一评测协议。
- 图：dataset taxonomy。

### 27. 训练数据集详解，1.5 分钟
- `GSV-Cities` / `GSV-Cities-light`。
- 规模、用途、训练稳定性。
- 为什么它适合训练全局检索模型。

### 28. 评测与定位数据集详解，2 分钟
- `Pitts30k-val` / `MSLS-val` / `Aachen Day-Night`。
- 各自对应什么场景：城市街景、跨季节/跨光照、重定位。
- 图：不同数据集场景图。

### 29. 训练与实现设置，1.5 分钟
- backbone、cluster 数、维度、Sinkhorn 步数。
- `mass_preserving`、学习率、batch size、loss 设定。
- 图：参数表。

### 30. 实验主结果 + 消融 + 总结，3 分钟
- 只放静态表格/曲线。
- 强调收敛速度、Recall、定位效果。
- 用一句话收束全文贡献。

## 备份页

### A1. 关键公式页
- OT 目标函数。
- Gaussian moments。
- Wasserstein residual。

### A2. 额外消融页
- OT / Gaussian query / refinement / mass preserving 的更多对比。
- 超参数稳定性图。

### A3. 数据集备份页
- VPR 数据集分类表。
- 各数据集适用任务与注意事项。

## 讲述策略

- 听众是混合背景，所以前 10 页会更偏直觉和任务定义。
- 方法页优先讲直觉，再给最关键公式。
- 代码只和方法对应，不展开实现细节。
- 实验只放静态图，不放视频或动图。

## 时间检查

- 总时长控制在 38-40 分钟。
- 每页讲解时长不超过 4 分钟。
- 现场优先保证“背景讲清楚、方法讲明白、结果讲得出结论”。

