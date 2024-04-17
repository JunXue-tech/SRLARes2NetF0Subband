from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_feeder import ASVDataSet, load_data
import feature_extract
from torch.autograd import Variable
from torch.nn import Parameter
from tqdm import tqdm
from model import SR_LA_Res2Net, SR_LA_block
import numpy as np
ids = [0]
from torch.utils.benchmark import Timer

train_protocol = "./ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trl.txt"
dev_protocol = "./ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
eval_protocol = "./ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval_test.trl.txt"



def SR_LA_Res2Net(**kwargs):
    model = SR_LA_Res2Net(SR_LA_block, [2, 2,2, 2], baseWidth=26, scale=8, **kwargs)
    return model

def feval(model, out_fold, feature_type, device):
    model.eval()
    name=os.path.join(out_fold, "eval.txt")
    result=[]
    scp=[]
    out_features = []
    labels = []
    label= []
    correct = 0
    timer = Timer()
    with torch.no_grad():
        wav_list, folder_list, flag, folder=load_data("eval", eval_protocol, mode="eval", feature_type="F0_subband")

        for idx in tqdm(range(len(wav_list)), desc="evaluating"):
            wav_id=wav_list[idx]
            scp.append(wav_id)
            label1 = folder_list[wav_id]
            wav_path = "{}{}.wav".format(folder, wav_id)
            feature=feature_extract.extract(wav_path, feature_type)
            data=np.reshape(feature, (-1, 1, 45, 600))
            data=torch.Tensor(data).to(device)
            output=model(data)
            result =output[0]
            label = np.array(label)
            label=torch.Tensor(label).to(device)
            label=label.unsqueeze(1)
            pred = result.argmax(dim=-1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()
        print('\nAccuracy: {}/{} ({:.0f}%)\n'.format(
                correct, len(wav_list),
                100. * correct / len(wav_list)))

            
class ScheduledOptim(object):
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 64
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.delta = 1

    def step(self):
        "Step by the inner optimizer"
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        "Learning rate scheduling per step"

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def state_dict(self):
        ret = {
            'd_model': self.d_model,
            'n_warmup_steps': self.n_warmup_steps,
            'n_current_steps': self.n_current_steps,
            'delta': self.delta,
        }
        ret['optimizer'] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        self.d_model = state_dict['d_model']
        self.n_warmup_steps = state_dict['n_warmup_steps']
        self.n_current_steps = state_dict['n_current_steps']
        self.delta = state_dict['delta']
        self.optimizer.load_state_dict(state_dict['optimizer'])

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='SR_LA_F0_subband')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=True, default='./models/')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for dev (default: 16)')
    parser.add_argument('--epochs', type=int, default=32, metavar='N',
                        help='number of epochs to train (default: 9)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--warmup', type=float, default=1000, metavar='M')
    #parser.add_argument('--keep_prob', type=float, default=1.0)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--feature_type', default='F0_subband')
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")


    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 2,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    model = SR_LA_Res2Net(num_classes=2).to(device)
    model.load_state_dict(torch.load(os.path.join(args.out_fold, "SR_LA_Res2Net.pt")))
    feval(model, args.out_fold, args.feature_type, device)

class A_softmax(nn.Module):
    def __init__(self, gamma=0):
        super(A_softmax, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte().bool()
        index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss

    if __name__ == '__main__':
        main()