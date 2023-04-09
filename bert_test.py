import torch
import torch.nn as nn
from transformers import BertConfig
from transformers import BertModel
from datasets import load_from_disk
import torch.optim as optim
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import BertConfig

# 定义计算设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test():
    model = CommentClassification()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.tensor([[1, 1, 1, 1]])
    token_type_ids = torch.tensor([[1, 1, 1, 1]])
    output = model(input_ids, attention_mask, token_type_ids)
    print(output)


# 训练函数
def train():

    # 构建模型
    # config = BertConfig()
    # config.num_hidden_layers = 2
    # config.num_attention_heads = 4
    # config.num_labels = 2
    # 微调
    # model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
    # 特征提取器
    # model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
    # 固定基础模型的参数
    # for param in model.base_model.parameters():
    #     param.requires_grad = False

    # 我们自己的模型使用的 4 层的隐层的模型，bert-base-chinese 是 12 层
    config = BertConfig()
    config.num_hidden_layers = 4
    config.num_labels = 2
    config.vocab_size = 21128
    # 使用 bert-base-chinese 前4个层的参数去初始化 BertForSequenceClassification 中 bertmodel 的4个层参数
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', config=config)


    # 加载数据
    train_data = load_from_disk('temp/data.comment')['train']
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化方法
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    # 训练轮次
    num_epochs = 10
    # 分词器
    tokenizer = BertTokenizer.from_pretrained('model/my-tokenizer')

    # 单步训练
    def train_step(review, label):
        # 对输入进行编码
        inputs = tokenizer(review, padding='longest', return_tensors='pt')
        # 将张量移动到指定设备上
        inputs = {name: value.to(device) for name, value in inputs.items()}
        # 将 label 转换为张量
        labels = torch.tensor(label, device=device)
        # 模型计算
        outputs = model(**inputs, labels=labels)

        # # 计算损失
        # label = torch.tensor(label, device=device)
        # loss = criterion(outputs, label)
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss = outputs.loss
        loss.backward()
        # 参数更新
        optimizer.step()

        nonlocal total_loss, total_iter
        total_loss += loss.item() * len(label)
        total_iter += len(label)

    for epoch in range(num_epochs):
        total_loss =0.0
        total_iter = 0
        train_data.map(train_step, input_columns=['review', 'label'], batched=True, batch_size=32, desc='Epoch %2d' % epoch)
        print('Loss: %.5f' % (total_loss / total_iter))
        # 模型存储
        torch.save(model.state_dict(), 'model/comment-classificaiton-%d.pt' % epoch)


if __name__ == '__main__':
    # train()

    params = torch.load('bert-base-chinese/pytorch_model.bin')
    print(params.keys())

    BertModel().embeddings
