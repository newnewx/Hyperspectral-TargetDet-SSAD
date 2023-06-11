import torch
import torch.nn as nn
import numpy as np

'''1.BCE'''
class BCE_Loss(nn.Module):
    def __init__(self,bcekwargs={}):
        super(BCE_Loss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss(**bcekwargs)

    def forward(self,net_output,target):

        # net_output = self.sigmoid(net_output)    #shape修改

        with torch.no_grad():
            if len(net_output.shape) != len(target.shape):
                target = target.view((target.shape[0], 1, *target.shape[1:]))

            if all([i == j for i, j in zip(net_output.shape, target.shape)]):
                # if this is the case then target is probably already a one hot encoding
                y_onehot = target
            else:
                target = target.long()
                y_onehot = torch.zeros(net_output.shape)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, target, 1)

        loss = self.bce(net_output, y_onehot.cuda())

        return loss
def test_bce():
    input = np.array([[ 1.4589, -1.4593],
                      [-0.8590,  0.2368]],dtype=np.float32).reshape((2,2,1,1))

    input = torch.from_numpy(input)
    #target = torch.Tensor([[0,1],[0,1]]).reshape((2,2,1,1))                  #taget：[b,c,h,w]
    target = torch.Tensor([1,1]).reshape((2,1,1))                             #taget：[b,h,w]

    bce = BCE_Loss(bcekwargs={'reduction':'none'})
    loss_bce = bce(input, target)
    print(loss_bce)

#test_bce()

'''结果'''
#tensor(1.0678)

