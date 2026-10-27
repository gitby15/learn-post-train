# Baseline设计

Baseline是用于检验模型的训练效果的，我们可以基于Baseline来定义：
- 模型的效果是否有提升
- 模型的泛化性是否有提升
- 模型的推理性能是否有提升
- 等等，核心是产生对比和基线


## Baseline设计
Baseline模型, 采用一个基座的Bert模型 + 一个线性分类头
这里选择用ModernBert + Linear 
损失函数：BCE with logits (在sigmoid之前做损失计算) 。 这样计算的话，数值更稳定


## 模型结构
ModernBERT-base + Shared Linear Head + BCEWithLogitsLoss
state = 'You are an idiot."
question = 'Is this text an insult?"
模型输入：
```python
[CLS]You are an idiot.[SEP]Is this text an insult?[SEP]
```

