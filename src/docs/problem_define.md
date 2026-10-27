# 问题定义

## 解决什么问题
我一直在纠结，是直接做一个简单的分类器，还是要层Laya的热度，做一个JEV风格的分类器。
虽然我知道，单纯的分类器，对与这个任务来说是最合适的（更简单、性能更好），但我最后还是下定决心，做一个JEV风格的分类器，一方面是可以学习更多的知识，另一方面是可以蹭Jev的热度，也许有助于我找下一份工作。

如果是单纯的分类器，那么问题定义就是：给定文本x，输出文本是有害信息的概率p
F(x) = p, p∈[ 0~1 ] # 0到1之间的概率

我们要实现的分类器，是给定文本x 和问题描述q，输出文本x符合描述q的概率p
F(x, q) = p, p∈[ 0~1 ] # 0到1之间的概率

```python
# laya还支持choice和score，为了简化我这次任务，这次我只支持noul类型
# 等我再学会一点，再进一步考虑choice和score
# 输入：
model.decide(
    state="You are an idiot.",
    questions={
        "insult": {
            "type": "noul",
            "instructions": "Is this text an insult?"
        },
        "threat": {
            "type": "noul",
            "instructions": "Does this text contain a threat?"
        }
    }
)

# 输出：针对不同语义，满足的概率
# 具体的输出格式参考下一个环节的设计结论
[0.88, 0.33]
```

## 能力边界：
- 语言：只支持英文
- 输入长度：小于512 token
- 问题类型：语义判断， 输入的state是否满足问题描述
- 任务领域：有害文本识别，包含侮辱(insult)、威胁(threat)、猥亵(obscene)等
- 支持的有害类型：insult、threat、obscene、identity_attack、sexual_explicit、toxicity、severe_toxicity
对于模糊、主观、需要大量外部背景知识的问题，不在我们要解决的范围内。

## 泛化能力
对于问题的泛化能力，分几个级别：
训练和测试使用相同任务、相同问题模板，但文本样本不同。

L1：基础能力泛化
> 相同任务、相同问题，但文本样本不同。

L2: 问题表达泛化
> 任务语义相同，但测试时使用训练中没有出现过的问题表述。
> 训练问题：Is this text an insult?
> 测试问题：Does this comment personally demean someone?


L3: 未见任务泛化
> 任务领域相同，但是测试时出现新的有害类型
> 比如说训练阶段没有使用「地域歧视识别」任务的监督标签，测试阶段首次要求模型判断文本是否包含地域歧视。

L4: 跨领域任务泛化
> 问了别的问题，比如说问了"do you like this comment?"
> 不在本次任务的范围内





