# 数据策略

## 概念
### 标签概念
首先要理解三个概念：
- 硬标签：0或者1，表示是否命中
- 软标签：0.8，可能是这表示10个人，有8个打了标签，也有可能是其他情况，不代表一定就是概率
- 模型预测概率：0.8，表示模型觉得有80%的概率是Yes
这三个语义是不一样的

### 类别不平衡
不同的标签，正例数量和负例数量的比例差异会很大
直接使用普通的BCE训练，会有一些问题：
- 假设threat的训练样本，正例的比例只有0.05%，那么模型见过的绝大部分样本，都是不带threat的样本，那么在测试时，模型总会倾向于预测较低的Threat概率
- 可能正例的数量、比例不够，导致模型没有学到正例的多样性（各种各样的Threat）

针对类别不平衡，我们该怎么做：
类别不平衡，不意味着模型训练失败，是否需要做重复采样、类别加权等操作，需要通过实验判断
- Weighted BCE（加权BCE损失）
- Oversampling (重复采样)
先不着急做这两个，可以先统计每个标签的Positive Rate

做三件事：
1. 判断数据集中不平衡的程度
2. 先训练原始分布的baseline - 这个很重要，如果baseline就很棒了，就没必要折腾，如果baseline不够好，就能作为优化的对照
3. Baseline表现不好，再考虑干预：Oversampling、Weighted BCE、Har Negative Mining


## 数据质量检查
You are an idiot.
you are an idiot!
You are  an idiot.
需要区分，这些非常相近的内容，语义是否相近，有时候标点符号不一样，但是语义是一样的，把他们当做不同的样本，会影响训练质量
1. 重复训练
2. 数据泄露(比如测试集也有这句话，只是多了个空格)

一个重要的工程原则：发现重复数据，不等于立刻删除重复数据，需要先弄清楚具体问题，比如说：
- 重复发生在哪里
- 重复程度
- 任务是否重复（是同一个标签吗）

Todo: 关于数据质量，应该是一门比较大的学问，但是我先入个门，在后面的学习过程中逐步深入了解

## 数据划分
需要解决的问题是：假设我们用Civil Comment的数据集，这个数据集要怎么处理，才能让L1,L2,L3的评估结果可信？
[隔离]
需要隔离两个维度：
- Text(文本)：模型是否在训练时见过相同的内容
- Task(任务)：模型是否在训练时接受过这类任务的监督（是否训练过这样的任务）

文本划分：
- 我们先按照官方划分，但是仍然需要过一道我们自己的重复检查，确保没有重复数据

任务划分：
假设选择其中一个Label作为L3的保留任务, 那么包含这个标签的样本，我们：
- Train 文本：非保留任务参与训练与 L1/L2 最终评估；保留任务标签用于 L3 最终评估。所有测试标签均不得用于训练、调参或模型选择
- Valiation 文本：其他的标签参与评估，参与调参。但是保留的标签不参与评估、调参与模型选择
- Test 文本：参与所有评估

对于提问的问题，我们准备不同的等价的问法：
- Train：Is this text an insult?
- Train：Does this text inslute someone?
- L2 Test: Does this language demean a person?
- L2 Test: Is someone being persoanlly attacked?
测试问法不能进入训练，也不能通过反复查看L2的Test成绩来选择模型或者调参
如果未来需要针对问法泛化调参，应该另外准备 L2 Validation 问法。


## Semantic Equivalence Check
语义等价性检查
|用途|	Question|
|Train 1|	Is this text an insult?|
|Train 2|	Does this text insult someone?|
|Train 3|	Does the text contain insulting language directed at someone?|  
|L2 Test 1|	Does this language demean a person or group?|
|L2 Test 2|	Is someone being personally attacked in this text?|

## 训练数据转换
假设civil comment的数据格式是：
```python
{
    "text": "You are an idiot.",
    "insult": 1,
    "threat": 0,
    "toxicity": 1
}
```
怎么把这个数据格式，转换成我们用于训练的格式? 
对于特定Label，设计一个问题map
```python
{
    "train1": 'Is this a xxx?',
    "train2": 'does any one have been xxx form this comment?',
    "train3": 'xxxx',
    "test1": 'xxxx',
    "test2": 'xxxx',
}
```
对于一个样本，做成如下格式：
```python
{
    "state": 'You are an idiot.',
    "insult": 0.8,
    "threat": 0,
    "obscene": 0.4
    ...
}
```

最终，训练数据的格式被定义成：
```python
example = {
    "state": "I will hurt you.",
    "task": "threat",
    "question": "Does this text contain a threat?",
    "label": 1,
}
```


## 数据验证
```python
example = {
    "state": "I will hurt you.",
    "task": "threat",
    "question": "Does this text contain a threat?",
    "label": 1,
}
def validate_example(example, allowed_tasks, templates):
    task = example["task"]

    assert task in allowed_tasks # 任务符合预期
    assert example["question"] in templates[task] # 问题不是空的
    assert example["label"] in (0, 1) # 标签值正常
    assert example["state"].strip() # 不是空的

    return True
```

