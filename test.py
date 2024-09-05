import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.benchmark import Timer
from tqdm import tqdm
from data_feeder import ASVDataSet, load_data
import feature_extract
from model import SR_LA_Res2Net, SR_LA_block

# 设置训练协议文件路径
train_protocol = "./ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trl.txt"
dev_protocol = "./ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
eval_protocol = "./ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval_test.trl.txt"

def create_model(**kwargs):
    """ 创建 SR_LA_Res2Net 模型 """
    return SR_LA_Res2Net(SR_LA_block, [2, 2, 2, 2], baseWidth=26, scale=8, **kwargs)

def feval(model, out_fold, feature_type, device):
    """ 模型评估函数 """
    model.eval()
    result_path = os.path.join(out_fold, "eval.txt")
    correct = 0
    wav_list, folder_list, flag, folder = load_data("eval", eval_protocol, mode="eval", feature_type=feature_type)

    with torch.no_grad():
        for idx in tqdm(range(len(wav_list)), desc="Evaluating"):
            wav_id = wav_list[idx]
            wav_path = f"{folder}{wav_id}.wav"
            feature = feature_extract.extract(wav_path, feature_type)
            data = torch.Tensor(np.reshape(feature, (-1, 1, 45, 600))).to(device)
            output = model(data)
            result = output[0]
            label = torch.Tensor([folder_list[wav_id]]).to(device)
            pred = result.argmax(dim=-1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(wav_list)
    print(f'\nAccuracy: {correct}/{len(wav_list)} ({accuracy:.0f}%)\n')

class ScheduledOptim:
    """ 学习率调度类 """
    def __init__(self, optimizer, n_warmup_steps, d_model=64):
        self.optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.d_model = d_model
        self.n_current_steps = 0
        self.delta = 1

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        self.n_current_steps += self.delta
        new_lr = (self.d_model ** -0.5) * min(
            self.n_current_steps ** -0.5, 
            self.n_warmup_steps ** -1.5 * self.n_current_steps
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def state_dict(self):
        state = {
            'd_model': self.d_model,
            'n_warmup_steps': self.n_warmup_steps,
            'n_current_steps': self.n_current_steps,
            'delta': self.delta,
            'optimizer': self.optimizer.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict):
        self.d_model = state_dict['d_model']
        self.n_warmup_steps = state_dict['n_warmup_steps']
        self.n_current_steps = state_dict['n_current_steps']
        self.delta = state_dict['delta']
        self.optimizer.load_state_dict(state_dict['optimizer'])

class A_softmax(nn.Module):
    """ 实现 A-Softmax 损失函数 """
    def __init__(self, gamma=0):
        super(A_softmax, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = self.LambdaMax

    def forward(self, input, target):
        self.it += 1
        cos_theta, phi_theta = input
        target = target.view(-1, 1)

        index = torch.zeros_like(cos_theta).scatter_(1, target.data.view(-1, 1), 1).byte()
        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        output = cos_theta.clone()
        output[index] -= cos_theta[index] * (1 + 0) / (1 + self.lamb)
        output[index] += phi_theta[index] * (1 + 0) / (1 + self.lamb)

        logpt = F.log_softmax(output, dim=1).gather(1, target).view(-1)
        pt = logpt.exp()
        loss = -1 * (1 - pt) ** self.gamma * logpt
        return loss.mean()

def main():
    """ 主程序 """
    parser = argparse.ArgumentParser(description='SR_LA_F0_subband Training')
    parser.add_argument("-o", "--out_fold", type=str, required=True, default='./models/', help="Output folder")
    parser.add_argument('--batch-size', type=int, default=64, help='Input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, help='Input batch size for dev (default: 64)')
    parser.add_argument('--epochs', type=int, default=32, help='Number of epochs to train (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
    parser.add_argument('--warmup', type=int, default=1000, help='Warmup steps for learning rate scheduling')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA training')
    parser.add_argument('--gpu', type=str, default="0", help="GPU index")
    parser.add_argument('--feature_type', default='F0_subband', help="Feature type for extraction")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = create_model(num_classes=2).to(device)
    try:
        model.load_state_dict(torch.load(os.path.join(args.out_fold, "SR_LA_Res2Net.pt")))
    except FileNotFoundError:
        print(f"Model not found in {args.out_fold}. Please check the path.")

    feval(model, args.out_fold, args.feature_type, device)

if __name__ == '__main__':
    main()
