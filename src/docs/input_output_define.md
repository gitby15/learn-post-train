# 输入输出定义

参考Laya的实现：
在API层，问题看似：F(x, [q1,q2,...]) -> [p1, p2,...]
在模型层，实际的解法是：F(x, q) -> p, 在计算的时候被合并成batch

## 输入和输出的格式

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
# state：需要审核的英文文本
# questions：一组自然语言问题描述
# question_id：问题的唯一标识符，用于在输出中引用问题描述，不参与计算
# type：问题的类型，这里只支持noul类型，返回yes的概率 | Laya还支持choice和score，这一期咱们先不考虑
# instructions：问题的描述，用于生成问题的提示



# 输出：针对不同语义，满足的概率
{
    "insult": 0.88,
    "threat": 0.33
}
# 返回类型：dict[str, float]
```

## 输入和输出的语义

输出 0.88 表示：给定 state 和对应的 instructions，模型预测该问题答案为 Yes 的概率，即
P(Y=1 | x,q)


## 异常输入处理
不合法的格式：
state为空、questions为空、输入不认识的question type，直接报错，不进行推理

输入格式合法，但是任务类型不在范围内
```python
model.decide(
    state="You are an idiot.",
    questions={
        "weather": {
            "type": "noul",
            "instructions": "Is it raining in London?"
        }
    }
)
```
不报错，暂时不考虑OOD问题


## 标签语义

Insult：文本中包含对个人或者群体的贬低、羞辱、或者人身攻击
- 针对人的贬低：属于
- 针对观点、行为、作品等：不属于
- 引用他人的言论：需要结合上下文判断，例如："The guy said that you are an idiot, he should not said that."

```
Threat：文本中表达了针对个人或者群体实施伤害、暴力或者其他严重不利行为的企图或威胁
- 你不xxx，我就要xxx(执行不利行为)：属于
- 描述客观的被伤害行为：不属于
- 引用他人的言论：需要结合上下文判断，例如："The bad guy in the movie said he will kill Tim"

Obscene: 包含粗俗、下流或冒犯性的语言，尤其是脏话
- Fuck your ass: 属于

Identity_attack: 因个人或群体的身份属性而进行贬低、敌视或攻击
- 你这个该死的Gay：属于
- 你是一个善良的Gay：不属于

Sexual_explicit: 明确描述性行为、性器官或露骨的性内容
- TBD

Toxicity: 具有攻击性、敌意或冒犯性，可能使交流环境变得不友善
- TBD

Severe_toxicity: 严重的敌意、侮辱、威胁或其他高度攻击性的表达
- TBD
```

标签之间不互斥
