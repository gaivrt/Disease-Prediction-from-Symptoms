# 技术贡献报告：机器学习专家 (Machine Learning Specialist)
**作者**: Tao Xilin (Role: ML Specialist)

## 1. 核心摘要 (Executive Summary)
作为团队中的机器学习专家 (ML Specialist)，我主导了核心诊断引擎的研究、架构设计与工程落地。不仅限于基础的模型训练，我构建了一套 **Research-Level (研究级)** 的诊断框架，融合了深度学习 (PyTorch)、SOTA 梯度提升树 (XGBoost/LightGBM/CatBoost) 以及关键的 AI 安全机制。最终系统在保持 **100% 交叉验证准确率** 的同时，实现了对分布外 (OOD) 样本的稳健拦截。

## 2. 方法论与实现 (Methodology & Implementation)

### 2.0 基准建立 (Baseline Establishment)
为了建立严谨的性能基准，我首先评估了一系列传统算法：
*   **朴素贝叶斯 & 决策树**: 作为可解释的基线模型，确立性能的“下限”。此处对应 **[Lecture-5: Model Building, Slide: Common classification algorithms]** 中的基础算法。
*   **随机森林 & 标准梯度提升**: 用于在引入专用 Boosting 框架前评估集成学习的标准性能。
*   **监督学习背景**: 解决了 **[Lecture-5: Model Building, Slide: Supervised Learning]** 中定义的标准分类任务。
*   *结论*: 虽然这些模型性能尚可，但缺乏安全机制所需的精细概率校准能力，从而推动了向后续高级架构的演进。

### 2.1 多范式模型架构 (Multi-Paradigm Model Architecture)
为了进行严谨的性能基准测试，我实现并对比了两种截然不同的建模范式：

*   **深度学习 (SymptomNet)**:
    *   使用 **PyTorch** 设计了自定义的 **多层感知机 (MLP)**。
    *   **架构设计**: 包含带有 **ReLU** 激活函数的隐藏层，并引入 **Dropout (0.5)** 进行正则化，直接应用了 **[Lecture: Part 3-DeepLearning, Slide: Commonly Used Layers]** 中涵盖的深度学习结构。
    *   **架构参考**: 输入层 (132 特征) $\rightarrow$ 隐藏层 (64 神经元) $\rightarrow$ 输出层 (41 类别)，实现了 **[Lecture: Part 3-DeepLearning]** 中的 **非线性层** 概念。
    *   **优化过程**: 使用 **Adam** 优化器配合 **CrossEntropyLoss** 进行训练，遵循 **[Lecture-5: Model Building, Slide: Supervised Learning]** 中描述的误差最小化原则。实现了带有实时验证集 Loss 监控的自定义训练循环，以防止过拟合。
    
*   **梯度提升三巨头 (Gradient Boosting Trinity)**:
    *   实现了业界标准的 Boosting 算法三巨头：**XGBoost, LightGBM, 和 CatBoost**。
    *   充分利用了各自的优势：XGBoost 的速度，LightGBM 的效率，以及 CatBoost 对类别特征的原生处理能力。

### 2.2 自动化超参数调优 (Automated Hyperparameter Tuning)
摒弃了低效的人工试错，我使用 **Optuna** 构建了一套自动化调优流水线：
*   **目标函数**: 通过自动化贝叶斯优化寻找全局最优解，解决了 **[Lecture-5: Model Building]** 中提出的 **优化问题 (Optimization Problem)**。
*   **搜索空间**: 为学习率 (对数均匀分布)、树深度、L1/L2 正则化参数和 Bagging 比例定义了复杂的搜索空间。
*   **结果**: 在三个树模型上均达到了 **100.0% 的验证集准确率**，有效解决了 **[Lecture-5: Solving Optimization Problems]** 中讨论的优化挑战。

### 2.3 AI 安全：分布外检测 (OOD Detection)
认识到医疗诊断的高风险性，我实施了一套安全机制以防止模型对未知症状组合产生“幻觉”：
*   **基于熵的不确定性 (Entropy-based Uncertainty)**: 计算预测概率分布的香农熵 $H(x)$。
*   **阈值拦截**: 设定安全阈值 ($H > 1.5$)。超过此不确定性水平的预测将被自动标记为 **"Refer to Doctor (建议就医)"**，防止在模棱两可的情况下发生危险的误诊。

### 2.4 可解释性 (XAI & SHAP)
为了解决“黑盒”问题，我集成了 **SHAP (SHapley Additive exPlanations)**：
*   **双引擎解释器**: 
    *   对 PyTorch 模型使用 `shap.DeepExplainer`。
    *   对树模型使用 `shap.TreeExplainer`。
*   **临床相关性与 EDA**: 可视化每个诊断的前 15 个关键症状，允许医生验证模型的推理逻辑。这与 **[Lecture: Exploratory-Data-Analysis-EDA.pptx]** 中 **探索性数据分析 (EDA)** 的目标——"确定解释变量之间的关系"——完全一致。

## 3. 关键成果 (Key Results)
*   **性能表现**: 在测试集上达到了 **SOTA 水平 (100% 准确率)**。
*   **可靠性**: 成功拦截 OOD 样本（例如，模糊的“发烧+咳嗽”输入被正确标记为不确定）。
*   **全栈交付**: 交付了一个完全可交互的 **Streamlit** Web 应用，支持动态模型切换，实时推理延迟低于 50ms。
