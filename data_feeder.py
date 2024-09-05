import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import feature_extract

def load_label(label_file):
    """
    加载标签文件并解析。

    参数：
    label_file (str): 标签文件的路径。

    返回：
    tuple: 包含标签字典和音频ID列表。
    - labels (dict): key 为音频ID，value 为标签 (0 表示 deepfake, 1 表示 bonafide)。
    - wav_lists (list): 所有音频文件的 ID 列表。
    """
    labels = {}
    wav_lists = []
    encode = {'deepfake': 0, 'bonafide': 1}
    
    with open(label_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                wav_id = line[2]
                wav_lists.append(wav_id)
                labels[wav_id] = encode[line[5]]
    
    return labels, wav_lists


class ASVDataSet(Dataset):
    """
    自定义数据集类，用于处理音频数据和标签。

    参数：
    data (list): 音频文件路径列表。
    label (list): 音频文件对应的标签列表。
    wav_ids (list, optional): 音频文件ID列表。
    transform (bool, optional): 是否进行数据转换，默认值为 True。
    mode (str, optional): 训练模式，可选 "train" 或 "eval"，默认值为 "train"。
    lengths (list, optional): 数据长度信息，默认值为 None。
    feature_type (str, optional): 特征提取方式，默认值为 "fft"。
    """

    def __init__(self, data, label, wav_ids=None, transform=True, mode="train", lengths=None, feature_type="fft"):
        super(ASVDataSet, self).__init__()
        self.data = data
        self.label = label
        self.wav_ids = wav_ids
        self.transform = transform
        self.lengths = lengths
        self.mode = mode
        self.feature_type = feature_type

    def __len__(self):
        """返回数据集的长度。"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取索引处的音频数据及其标签，并进行特征提取。

        参数：
        idx (int): 数据集中的索引。

        返回：
        tuple: 包含处理后的数据和对应标签。
        """
        each_data, each_label = self.data[idx], self.label[idx]
        each_data = feature_extract.extract(each_data, self.feature_type)
        
        if self.transform:
            each_data = torch.Tensor(each_data)
        
        return each_data, each_label


def load_data(dataset, label_file, train_data_path, dev_data_path, eval_data_path, mode="train", feature_type="fft"):
    """
    根据模式加载数据。

    参数：
    dataset (str): 数据集名称。
    label_file (str): 标签文件路径。
    train_data_path (str): 训练数据文件路径（作为变量传入）。
    dev_data_path (str): 验证数据文件路径（作为变量传入）。
    eval_data_path (str): 评估数据文件路径（作为变量传入）。
    mode (str): 加载模式，"train" 或 "eval"。
    feature_type (str): 特征提取方式。

    返回：
    tuple: 根据模式返回不同的数据和标签信息。
    """
    if mode != "eval":
        return load_train_data(dataset, label_file, train_data_path, dev_data_path, feature_type)
    else:
        data, folder_list, flag = load_eval_data(dataset, label_file, eval_data_path, feature_type)
        return data, folder_list, flag, eval_data_path


def load_train_data(dataset, label_file, train_data_path, dev_data_path, feature_type="fft"):
    """
    加载训练数据并提取特征。

    参数：
    dataset (str): 数据集名称。
    label_file (str): 标签文件路径。
    train_data_path (str): 训练数据文件路径。
    dev_data_path (str): 验证数据文件路径。
    feature_type (str): 特征提取方式。

    返回：
    tuple: 包含音频文件路径列表和对应的标签列表。
    """
    labels, wav_lists = load_label(label_file)
    final_data = []
    final_label = []

    for wav_id in tqdm(wav_lists, desc=f"Loading {dataset} data"):
        label = labels[wav_id]

        if "_T_" in wav_id.upper():
            wav_path = os.path.join(train_data_path, f"{wav_id}.flac")
        elif "_D_" in wav_id.upper():
            wav_path = os.path.join(dev_data_path, f"{wav_id}.flac")
        else:
            print(f"Unknown identifier in {wav_id}, skipping.")
            continue

        if os.path.exists(wav_path):
            final_data.append(wav_path)
            final_label.append(label)
        else:
            print(f"Cannot open {wav_path}")
    
    return final_data, final_label


def load_eval_data(dataset, scp_file, eval_data_path, feature_type="fft"):
    """
    加载评估数据。

    参数：
    dataset (str): 数据集名称。
    scp_file (str): 数据文件路径。
    eval_data_path (str): 评估数据文件路径。
    feature_type (str): 特征提取方式。

    返回：
    tuple: 包含音频文件ID列表、文件夹列表和标识符信息。
    """
    wav_lists = []
    folder_list = {}
    flag = {}

    with open(scp_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                wav_id = line[1]
                wav_lists.append(wav_id)
                folder_list[wav_id] = line[-1]
                flag[wav_id] = line[-2] if line[-2] != '-' else 'A00'

    return wav_lists, folder_list, flag


def main():
    """
    主函数，测试标签加载和部分音频文件ID及标签的打印。

    参数：
    - train_data_path: 训练数据的文件夹路径
    - dev_data_path: 开发数据的文件夹路径
    - eval_data_path: 评估数据的文件夹路径
    """
    train_data_path = "path/to/train_data"  # 训练数据路径
    dev_data_path = "path/to/dev_data"  # 开发数据路径
    eval_data_path = "path/to/eval_data"  # 评估数据路径
    label_file = "path/to/label_file.txt"  # 标签文件路径

    labels, wav_lists = load_label(label_file)
    wav_list = wav_lists[4000:4005]
    
    for wav_id in wav_list:
        print(f"Audio ID: {wav_id}")
        print(f"Label: {labels[wav_id]}")


if __name__ == '__main__':
    main()
