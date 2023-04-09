import pandas as pd
import torch
from datasets import Dataset
from datasets import DatasetDict
from pyhanlp import JClass



def load_data():

    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')

    # 1. 对样本的标签构建编码
    labels = []
    for label in train_data['label'].tolist():
        labels.extend(label.split('|'))

    # 标签去重
    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels.append(label)

    # [0, 0, 0, 1, 1, 0, 0....]  65长度
    # 给每一个标签分配一个唯一的编号
    label_to_index = { label: index for index, label in enumerate(unique_labels)}
    torch.save(label_to_index, 'data/label-index.data')

    # 2. 对输入文本进行标准化，标签编码
    # 先将 train_data test_data 转换为 Dataset 对象
    train_data = Dataset.from_pandas(train_data)
    test_data = Dataset.from_pandas(test_data)
    # 再将两个 Dataset 对象转换成 DatasetDict 对象
    datasets = DatasetDict({'train': train_data, 'test': test_data})


    def data_handler(label, content):

        # 内容需要标准化
        normalizer = JClass('com.hankcs.hanlp.dictionary.other.CharTable')
        content = normalizer.convert(content)

        # 标签需要编码
        muti_hot = [0.0] * len(label_to_index)
        label_index = [ label_to_index[la] for la in label.split('|')]
        for index in label_index:
            muti_hot[index] = 1.0

        return {'label': muti_hot, 'content': content}


    datasets = datasets.map(data_handler, input_columns=['label', 'content'])

    # 存储处理好的数据
    datasets.save_to_disk('data/train-test.data')


if __name__ == '__main__':
    load_data()