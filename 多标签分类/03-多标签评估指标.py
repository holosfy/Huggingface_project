import torch


y_true = torch.tensor([[0 ,0, 0, 1, 1]])
# 假设：我们将一个样本送入到模型，最终模型会输出 logits
logits = torch.tensor([[0.15, -0.34, 0.12, -0.67, 0.55]])
# 每个样本的预测概率
proba = torch.sigmoid(logits)
# 定义一个阈值，只要概率大于该阈值，该样本就有可能属于该标签，假设：0.5
y_pred = torch.where(proba > 0.5, 1, 0)

print('y_true:', y_true)
print('y_pred:', y_pred)

def test():

    # 预测正确的标签数量
    y_correct = torch.sum(y_true & y_pred)

    # 1. 计算单个样本的准确率
    acc = y_correct / torch.sum(y_true | y_pred)
    print('准确率:', acc)

    # 2. 计算单个样本的精确率
    pre = y_correct / torch.sum(y_pred)
    print('精确率:', pre)

    # 3. 计算单个样本的召回率
    rec = y_correct / torch.sum(y_true)
    print('召回率:', rec)

    # 4. 计算单个样本的f1-score
    f1c = (2 * y_correct) / (torch.sum(y_true) + torch.sum(y_pred))
    print('f1值:', f1c)


if __name__ == '__main__':
    test()
