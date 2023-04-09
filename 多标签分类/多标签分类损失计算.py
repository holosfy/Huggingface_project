import torch.nn as nn
import torch


# 1. 单标签分类损失计算
# 假设: 真实标签 2，预测分数 [0.15, -0.34, 0.12, 0.67, 0.55]
def test01():

    labels = torch.tensor([2])
    logits = torch.tensor([[0.15, -0.34, 0.12, 0.67, 0.55]])

    # logits -> softmax ->
    loss = nn.CrossEntropyLoss()(logits, labels)
    print(loss)

    # 慢动作
    # 1. 对每个 logit 计算指数
    temp = torch.exp(logits)

    # 2. 计算每个 logit 概率值
    probas = temp / torch.sum(temp, dim=-1)

    # 3. 计算正确标签对应概率值的负对数值
    loss = -torch.log(probas[0][2])
    print(loss)


    # CrossEntropyLoss 内部会对输入的 logits 先进行 softmax 计算，然后再计算负对数损失
    # 对于单标签分类，如果我们把标签使用 one hot 表示，形式应该是 [0, 0, 1, 0, 0]


# 2. 多标签分类损失计算
# 假设：某个样本的标签 [1, 3]，multi hot [0, 1, 0, 1, 0]
# 模型输出当前样本属于某个标签的分数： [0.15, -0.34, 0.12, 0.67, 0.55]
# 把每个样本的多个标签理解为独立(不互斥)，把每一个标签当做独立的二分类问题
def test02():

    logits = torch.tensor([[0.15, -0.34, 0.12, 0.67, 0.55]])
    # 计算多标签损失时，真实标签需要使用 multi-hot 形式表示，必须是 float32 类型
    labels = torch.tensor([[0, 1, 0, 1, 0]], dtype=torch.float32)

    # 在 pytorch 中有三个 API 都可以进行多标签损失计算
    # BCELoss、BCELossWithLogits、MultiLabelSoftMarginLoss
    # 1. BCELoss 计算多标签损失，传入的 logits 需要先进行 sigmoid 计算
    loss = nn.BCELoss()(torch.sigmoid(logits), labels)
    print(loss)

    # 2. BCELossWithLogits 计算多标签的损失
    loss = nn.BCEWithLogitsLoss()(logits, labels)
    print(loss)

    # 3. MultiLabelSoftMarginLoss 计算多标签损失
    loss = nn.MultiLabelSoftMarginLoss()(logits, labels)
    print(loss)


    # 慢动作分解
    # 1.先对 1 位置对应的分数使用 sigmoid 函数，计算其为 1 的概率；
    temp = torch.sigmoid(logits)
    print(temp)

    # 2. 对 1 类别分数计算 -torch.log(logit_sigmoid), 对 0 类别分数计算 -torch.log(1-logit_sigmoid)
    loss = 0.0
    for index, value in zip([0, 1, 0, 1, 0], temp[0]):
        if index == 0:
            loss += -torch.log(1-value)
        if index == 1:
            loss += -torch.log(value)

    # 3. 将负对数值相加再计算均值即可得到某个样本多标签的损失值。
    loss = loss / 5
    print(loss)


if __name__ == '__main__':
    # test01()
    test02()