import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from datasets import load_from_disk
import torch.optim as optim

# 定义计算设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():

    # 1. 构建必要的对象
    # 加载 label 编码字典
    label_index = torch.load('data/label-index.data')
    # 构建模型 BertForSequenceClassification
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=len(label_index))
    model = model.to(device)
    # 分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # 训练数据
    train_data = load_from_disk('data/train-test.data')['train']
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)


    # 2. 开始训练
    def train_step(label, content):
        # 输入编码
        inputs = tokenizer(content, padding='longest', return_tensors='pt')
        inputs = {name: value.to(device) for name, value in inputs.items()}
        labels = torch.tensor(label, device=device)
        # 模型计算
        outputs = model(**inputs, labels=labels)
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss = outputs.loss
        loss.backward()
        # 参数更新
        optimizer.step()

        # 训练信息
        nonlocal total_loss, total_iter
        total_loss += loss.item() * len(labels)
        total_iter += len(labels)

    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_iter = 0
        # shuffle 将数据集打乱
        train_data.shuffle().map(train_step, batched=True, batch_size=8, input_columns=['label', 'content'], desc='Epoch %2d' % epoch)
        print('Loss: %.5f' % (total_loss / total_iter))
        # 模型存储
        model.save_pretrained('model')
        tokenizer.save_pretrained('model')


if __name__ == '__main__':
    train()