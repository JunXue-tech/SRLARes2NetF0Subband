import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_feeder import ASVDataSet, load_data
from model import SR_LA_Res2Net, SR_LA_block
from torch.utils.benchmark import Timer
import numpy as np

# 设置训练协议文件路径
train_protocol = "./ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trl.txt"
dev_protocol = "./ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
eval_protocol = "./ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval_test.trl.txt"

def create_model(**kwargs):
    """创建 SR_LA_Res2Net 模型"""
    return SR_LA_Res2Net(SR_LA_block, [2, 2, 2, 2], baseWidth=26, scale=8, **kwargs)

class A_softmax(nn.Module):
    """实现 A-Softmax 损失函数"""
    def __init__(self, gamma=0):
        super(A_softmax, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta, phi_theta = input
        target = target.view(-1, 1)  # size=(B,1)

        index = torch.zeros_like(cos_theta).scatter_(1, target.data.view(-1, 1), 1).byte()
        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        output = cos_theta.clone()
        output[index] -= cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
        output[index] += phi_theta[index] * (1.0 + 0) / (1 + self.lamb)

        logpt = F.log_softmax(output, dim=1).gather(1, target).view(-1)
        pt = logpt.exp()
        loss = -1 * (1 - pt) ** self.gamma * logpt
        return loss.mean()

def train(args, model, device, train_loader, optimizer, epoch):
    """训练函数"""
    model.train()
    correct = 0
    criterion = A_softmax()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        result = output[0]
        pred = result.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        if args.dry_run:
            break

def test(model, device, test_loader):
    """测试函数"""
    model.eval()
    test_loss = 0
    correct = 0
    criterion = A_softmax()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            result = output[0]
            pred = result.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.6f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return test_loss

class ScheduledOptim:
    """学习率调度类"""
    def __init__(self, optimizer, n_warmup_steps, d_model=64):
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
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
        return {
            'd_model': self.d_model,
            'n_warmup_steps': self.n_warmup_steps,
            'n_current_steps': self.n_current_steps,
            'delta': self.delta,
            'optimizer': self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.d_model = state_dict['d_model']
        self.n_warmup_steps = state_dict['n_warmup_steps']
        self.n_current_steps = state_dict['n_current_steps']
        self.delta = state_dict['delta']
        self.optimizer.load_state_dict(state_dict['optimizer'])

def main():
    """主程序"""
    parser = argparse.ArgumentParser(description='SR_LA_F0_subband Training')
    parser.add_argument("-o", "--out_fold", type=str, required=True, default='./models/', help="Output folder")
    parser.add_argument('--batch-size', type=int, default=64, help='Input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, help='Input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=32, help='Number of epochs to train (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
    parser.add_argument('--warmup', type=int, default=1000, help='Warmup steps for learning rate scheduling')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False, help='Quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, help='How often to log training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For saving the current model')
    parser.add_argument('--feature_type', default='f0_subband', help='Feature type for extraction')
    parser.add_argument("--gpu", type=str, default="0", help="GPU index")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 创建模型输出文件夹
    if not os.path.exists(args.out_fold):
        os.makedirs(args.out_fold)

    # 加载数据
    train_data, train_label = load_data("train", train_protocol, mode="train")
    train_dataset = ASVDataSet(train_data, train_label, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    dev_data, dev_label = load_data("dev", dev_protocol, mode="train")
    dev_dataset = ASVDataSet(dev_data, dev_label, mode="train")
    dev_loader = DataLoader(dev_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = create_model(num_classes=2).to(device)
    optimizer = ScheduledOptim(optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        args.warmup)

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        loss = test(model, device, dev_loader)

        # 保存最优模型
        if args.save_model and loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), os.path.join(args.out_fold, 'SR_LA_Res2Net.pt'))
            print("Model saved")

if __name__ == '__main__':
    main()
