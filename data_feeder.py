import numpy as np
#import librosa
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
#import scipy.io as scio
#from sklearn.preprocessing import scale
import feature_extract
import os

def load_label(label_file):
    labels = {}
    wav_lists = []
    #folder_list={}
    encode = {'deepfake': 0, 'bonafide': 1}
    with open(label_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                wav_id = line[2]
                wav_lists.append(wav_id)
                #wav_id = line[1].replace(".wav", "")
                tmp_label = encode[line[5]]
                labels[wav_id] = tmp_label
                #folder_list[wav_id]=line[0]
    return labels, wav_lists#, folder_list


class ASVDataSet(Dataset):

    def __init__(self, data, label, wav_ids=None, transform=True, mode="train", lengths=None, feature_type="fft"):
        super(ASVDataSet, self).__init__()
        self.data = data
        self.label = label
        self.wav_ids = wav_ids
        self.transform = transform
        self.lengths = lengths
        self.mode = mode
        self.feature_type=feature_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        each_data, each_label = self.data[idx], self.label[idx]
        each_data=feature_extract.extract(each_data,self.feature_type)
        if self.transform:
            #each_data, each_label = torch.from_numpy(each_data).float(), torch.LongTensor([each_label])
            #each_data = torch.from_numpy(each_data).float()
            each_data=torch.Tensor(each_data)
        return each_data, each_label


# this will load data by wav
def load_data(dataset, label_file, mode="train", feature_type="fft"):
    if mode!="eval":
        data, label=load_train_data(dataset, label_file, feature_type="fft")
        return data,label
    else:
        #data, folder=load_eval_data(dataset, label_file, feature_type="fft")
        data, folder_list, flag = load_eval_data(dataset, label_file, feature_type="fft")
        folder = "G:\Dataset\ASVspoof2019data\ASVspoof2019_LA_eval/"
        return data, folder_list, flag, folder


def load_train_data(dataset, label_file, feature_type="fft"):
    #labels, wav_lists, folder_list = load_label(label_file)
    labels, wav_lists = load_label(label_file)
    final_data = []
    final_label = []

    for wav_id in tqdm(wav_lists, desc="load {} data".format(dataset)):
        #wav_id = wav_name.replace(".wav", "")
        label = labels[wav_id]
        #folder = folder_list[wav_id]
        
        if "_T_" in wav_id.upper():
            wav_path = "G:/SVDD/SVDD_dataset/train_set/{}.flac".format(wav_id)
        if "_D_" in wav_id.upper():
            wav_path = "G:/SVDD/SVDD_dataset/dev_set/{}.flac".format(wav_id)
        
        #wav_path = "../data/ASVspoof2015/wav/{}/{}.wav".format(folder, wav_id)
        #feature=feature_extract.extract(wav_path, feature_type)
        if os.path.exists(wav_path):
            final_data.append(wav_path)
        
            #for j in range(feature.shape[1]/400):
            final_label.append(label)
        else:
            print("can not open {}".format(wav_path))
        
        #final_wav_ids.append(wav_id)
    
    #final_data=np.concatenate(final_data,axis=1)

    #final_data = np.reshape(np.array(final_data),(-1,1,864,400))
    return final_data, final_label

def load_eval_data(dataset, scp_file, feature_type="fft"):
    wav_lists = []
    folder_list={}
    flag = {}
    with open(scp_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                wav_id = line[1]
                wav_lists.append(wav_id)
                folder_list[wav_id]=line[-1]
                if line[-2] == '-':
                    flag[wav_id] = 'A00'
                else:
                    flag[wav_id] = line[-2]
    return wav_lists, folder_list, flag

def main():
    labels, wav_lists=load_label("ASVspoof2019.LA.cm.train.trl.txt")
    wav_list=wav_lists[4000:4005]
    for wav_id in wav_list:
        print(wav_id)
        print(labels[wav_id])

if __name__ == '__main__':
    main()
