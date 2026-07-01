# 36页VPR组会PPT重设计方案

目标：把原来的 16 页扩展成 **36 页**，仍然控制在 **40 分钟** 内，且内容完全落在 **VPR（Visual Place Recognition）** 上，不再强调分层定位或系统管线。新版更适合“混合背景”听众，也方便把重点放在“VPR 是什么、为什么重要、当前方法怎么做、数据集怎么选、实验怎么证明有效”。

总节奏建议：**背景 8 分钟 + 相关工作 5 分钟 + 方法 15 分钟 + 数据集 8 分钟 + 实验/总结 4 分钟**

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
- 图：query -> retrieve 的简图。

### 4. VPR 为什么重要，1 分钟
- VPR 是大规模视觉检索与场景理解的基础环节。
- 它常作为重定位、导航、地图检索等下游任务的候选生成器。
- 图：VPR -> downstream tasks 示意图。

### 5. VPR 的应用场景，1 分钟
- 城市街景检索。
- 长时序环境识别。
- 跨季节、跨光照、跨视角匹配。
- 图：不同应用场景拼图。

### 6. 为什么“检索质量”是上限，1.5 分钟
- 检索错了，后面的匹配和判别很难补回来。
- VPR 评测里，Recall@K 往往直接决定可用性。
- 图：retrieval quality -> final success 的链路图。

### 7. 背景小结，1 分钟
- 统一问题：全局检索要快，还要稳。
- 为后面方法动机做转场。

### 8. VPR 方法演进图，1 分钟
- 从 NetVLAD 到 BOQ / SALAD / QAA / SELA 等趋势。
- 先给“方法谱系”，不展开公式。
- 图：方法族谱或时间线。

### 9. NetVLAD 的局限，1 分钟
- 一阶聚合只能表达“加权和”。
- 对局部结构、分布和不确定性刻画不足。
- 图：VLAD 聚合示意。

### 10. 最近 VPR 趋势：BOQ / SALAD / QAA / SELA，1.5 分钟
- 强调最新工作在“token 选择、质量建模、聚合方式”上的方向。
- 只讲趋势和启发，不做论文综述。
- 图：对比表。

### 11. 现有方法痛点总结，1 分钟
- 一阶聚合不够。
- token 质量和全局约束没被显式建模。
- 这就是本文切入点。

### 12. 论文贡献概览，1.5 分钟
- OT 分配。
- 高斯残差描述子。
- Wasserstein 几何解释。
- OT-anchored refinement。

### 13. 整体方法框架图，1.5 分钟
- backbone -> OT assignment -> moments -> residual descriptor -> refinement。
- 图：`ot_gaussvlad2_arch.pdf` 或重绘总图。

### 14. Backbone 到 token，1 分钟
- 输入特征图如何被映射成局部 tokens。
- 为什么需要聚合前的维度压缩。
- 图：backbone + adapter。

### 15. A：OT 运输分配直觉，1.5 分钟
- score matrix、dustbin、全局质量约束。
- 核心点：不是 token-wise softmax。
- 图：运输矩阵示意。

### 16. A：Sinkhorn 与 dustbin，1.5 分钟
- 行列边际约束。
- dustbin 吸收低质量 token。
- 图：mass flow 图。

### 17. A：数学表达，0.5 分钟
- \(A^\star = \mathrm{diag}(u)K\mathrm{diag}(v)\)
- 只保留关键公式和直觉。

### 18. A：transport mass 与 token saliency，1 分钟
- 讲 token mass 的作用。
- 说明为什么需要质量分配而不是均匀平均。

### 19. B：Transport-induced Gaussian moments，1.5 分钟
- \(\hat{\mu}_k\)、\(\hat{v}_k\)、\(\hat{\sigma}_k\) 的定义。
- 它们是“当前图像里这个簇的统计量”。
- 图：cluster mass -> moment estimation。

### 20. B：moment 的直觉解释，1 分钟
- 均值是“中心在哪里”。
- 方差/标准差是“这个簇有多散”。
- 图：mean / variance 可视化。

### 21. C：Wasserstein residual descriptor，1.5 分钟
- 残差怎么从 \(\hat{\mu}_k,\hat{\sigma}_k\) 和 prototype 生成。
- 为什么比普通 VLAD 更强。
- 图：mean/std residual 拼接图。

### 22. C：为什么和 diagonal-Gaussian 几何一致，1 分钟
- 对角高斯下，Wasserstein 距离可化成欧氏形式。
- 这让 descriptor 既有几何解释又便于学习。

### 23. D：OT-anchored refinement 动机，1 分钟
- refinement 不是替代 OT，而是修正 OT 统计。
- 为什么“先 OT，再修正”更稳。

### 24. D：refiner 结构图，1.5 分钟
- `LayerNorm -> Linear -> GELU -> Dropout -> Linear`
- `MultiheadAttention`
- 可选 FFN。
- 图：refiner 模块图。

### 25. D：门控与初始化细节，1 分钟
- 零初始化 head。
- `inverse_sigmoid` 小门控。
- `max_residual_scale` 约束。
- 讲清“从接近基线开始学”。

### 26. 代码和论文的对应关系，0.5 分钟
- 只讲高层对应，不展开实现细节。
- 把 `OT`, `moments`, `refinement` 三块和代码模块对上。

### 27. 数据集总览：为什么要单独讲，1 分钟
- VPR 数据集类型很多，场景、标注和划分方式不统一。
- 需要标准化 downloader 和统一评测协议。
- 图：dataset taxonomy。

### 28. 训练数据集：GSV-Cities 系列，1.5 分钟
- `GSV-Cities` / `GSV-Cities-light` 是训练主力。
- 适合学城市街景中的地点不变性。
- 讲清楚轻量版和完整版的差别。

### 29. 验证数据集：Pitts30k / Pitts250k / MSLS，1.5 分钟
- `Pitts30k-val`、`Pitts250k`、`MSLS-val` 都是常用检索基准。
- 重点看 Recall@K、召回稳定性和跨视角鲁棒性。
- 说明为什么这类数据最适合先验证检索器。

### 30. 城市场景测试集：San Francisco / Eynsham / Tokyo 24/7，1.5 分钟
- `San Francisco Landmark` 体现城市地标检索。
- `Eynsham` 体现长期变化下的地点识别。
- `Tokyo 24/7` 体现昼夜和视角变化。

### 31. 长时序与季节变化：St Lucia / Nordland / SVOX / AmsterTime，1.5 分钟
- `St Lucia` 是经典 suburb 场景。
- `Nordland` 体现强季节变化。
- `SVOX`、`AmsterTime` 体现跨时段与长时序鲁棒性。

### 32. 更难的检索场景：SPED / Baidu / Mapillary SLS，1.5 分钟
- `SPED`、`Baidu` 代表更大规模或更复杂的城市环境。
- `Mapillary SLS` 覆盖街景多样性，适合检验泛化能力。
- 强调这些数据更接近真实部署难度。

### 33. 数据集选择原则与评测协议，1 分钟
- 训练集、验证集、测试集要分开。
- 先做检索 Recall，再看 retrieval robustness。
- 评测时区分验证集上的快速筛选与测试集上的最终报告。

### 34. 训练与实现设置，1.5 分钟
- backbone、cluster 数、维度、Sinkhorn 步数。
- `mass_preserving`、学习率、batch size、loss 设定。
- 图：参数表。

### 35. 实验主结果，2 分钟
- 只放静态表格/曲线。
- 强调收敛速度、Recall、鲁棒性。
- 图：训练损失曲线 + recall 表。

### 36. 消融分析与总结，2 分钟
- OT / Gaussian query / refinement / mass preserving 的贡献。
- 用一句话收束全文贡献。
- 图：ablation 表格 + take-home message。

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
