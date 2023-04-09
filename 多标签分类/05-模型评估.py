import torch
from datasets import load_from_disk
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate():

    # 1. 加载测试集数据
    test_data = load_from_disk('data/train-test.data')['test']
    # 2. 加载分词器
    tokenizer = BertTokenizer.from_pretrained('model')
    # 3. 加载模型
    model = BertForSequenceClassification.from_pretrained('model').to(device)
    model.eval()
    # 4. 开始预测
    accuracy_list = []
    precision_list = []
    recall_list = []
    def evaluate_step(label, content):

        # 输入编码
        inputs = tokenizer(content, padding='longest', return_tensors='pt')
        inputs = { name: value.to(device) for name, value in inputs.items() }
        y_true = torch.tensor(label, dtype=torch.int64)

        with torch.no_grad():

            # 模型计算
            outputs = model(**inputs)

            # 分数转换标签
            proba = torch.sigmoid(outputs.logits)
            y_pred = torch.where(proba >= 0.5, 1, 0).cpu()

            # 计算 y_true 和 y_pred 的交集, 模型预测正确的标签数量
            right = torch.sum(y_true & y_pred, dim=-1)
            # 计算 y_true 和 y_pred 的并集
            union = torch.sum(y_true | y_pred, dim=-1)
            # 预测标签数量
            y_pred = torch.sum(y_pred, dim=-1)
            # 真实标签数量
            y_true = torch.sum(y_true, dim=-1)

            # 1. 计算准确率
            accuracy = right / union
            accuracy_list.extend(accuracy.numpy())

            # 2. 计算精确率
            precision = right / y_pred
            # y_pred 有没有可能为 0 ？
            precision = torch.where(torch.isnan(precision), torch.tensor(0.0), precision)
            precision_list.extend(precision.numpy())

            # 3. 计算召回率
            recall = right / y_true
            recall_list.extend(recall.numpy())

    test_data.map(evaluate_step, batched=True, batch_size=8, input_columns=['label', 'content'])

    # 5. 打印评估指标
    print('准确率:', np.mean(accuracy_list))
    print('精确率:', np.mean(precision_list))
    print('召回率:', np.mean(recall_list))


if __name__ == '__main__':
    evaluate()