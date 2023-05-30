# ==============================================================================
# Copyright <2019> <Chen Wang [https://chenwang.site], Carnegie Mellon University>
# Refer to: https://github.com/wang-chen/interestingness/blob/tro/torchutil.py
# ==============================================================================

import cv2
import time
import math
import torch
import random
import torch.fft
import collections
import torchvision
from torch import nn
from itertools import repeat
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def tic(self):
        self.start()

    def show(self, prefix="", output=True):
        duration = time.time()-self.start_time
        if output:
            print(prefix+"%fs" % duration)
        return duration

    def toc(self, prefix=""):
        self.end()
        print(prefix+"%fs = %fHz" % (self.duration, 1/self.duration))
        return self.duration

    def start(self):
        torch.cuda.synchronize()
        self.start_time = time.time()

    def end(self):
        torch.cuda.synchronize()
        self.duration = time.time()-self.start_time
        self.start()
        return self.duration


class MovAvg(nn.Module):
    def __init__(self, window_size=3):
        super(MovAvg, self).__init__()
        assert(window_size>=1)
        self.window_size = window_size
        weight = torch.arange(1, window_size+1).type('torch.FloatTensor')
        self.register_buffer('weight', torch.zeros(1,1,window_size))
        self.weight.data = (weight/weight.sum()).view(1,1,-1)
        self.nums = []

    def append(self, point):
        if len(self.nums) == 0:
            self.nums = [point]*self.window_size
        else:
            self.nums.append(point)
            self.nums.pop(0)
        return F.conv1d(torch.tensor(self.nums, dtype=torch.float).view(1,1,-1), self.weight).view(-1)


class ConvLoss(nn.Module):
    def __init__(self, input_size, kernel_size, stride, in_channels=3, color=1):
        super(ConvLoss, self).__init__()
        self.color, input_size, kernel_size, stride = color, _pair(input_size), _pair(kernel_size), _pair(stride)
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=kernel_size, stride=stride, bias=False)
        self.conv.weight.data = torch.ones(self.conv.weight.size()).cuda()/self.conv.weight.numel()
        self.width = (input_size[0] - kernel_size[0]) // stride[0] + 1
        self.hight = (input_size[0] - kernel_size[1]) // stride[1] + 1
        self.pool = nn.MaxPool2d((self.width, self.hight))

    def forward(self, x, y):
        loss = self.conv((x-y).abs())
        value, index = loss.view(-1).max(dim=0)
        w = (index//self.width)*self.conv.stride[0]
        h = (index%self.width)*self.conv.stride[1]
        x[:,:,w:w+self.conv.kernel_size[0],h] -= self.color
        x[:,:,w:w+self.conv.kernel_size[0],h+self.conv.kernel_size[1]-1] -= self.color
        x[:,:,w,h:h+self.conv.kernel_size[1]] -= self.color
        x[:,:,w+self.conv.kernel_size[0]-1,h:h+self.conv.kernel_size[1]] -= self.color
        return value


class CosineSimilarity(nn.Module):
    '''
    Averaged Cosine Similarity for 3-D tensor(C, H, W) over channel dimension
    Input Shape:
    x: tensor(N, C, H, W)
    y: tensor(B, C, H, W)
    Output Shape:
    o: tensor(N, B)
    '''
    def __init__(self, eps=1e-7):
        super(CosineSimilarity, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        N, C, H, W = x.size()
        B, c, h, w = y.size()
        assert(C==c and H==h and W==w)
        x, y = x.view(N,1,C,H*W), y.view(B,C,H*W)
        xx, yy = x.norm(dim=-1), y.norm(dim=-1)
        xx[xx<self.eps], yy[yy<self.eps] = self.eps, self.eps
        return ((x*y).sum(dim=-1)/(xx*yy)).mean(dim=-1)


class CosineLoss(nn.CosineEmbeddingLoss):
    def __init__(self, dim=1):
        super(CosineLoss, self).__init__()
        self.target = torch.ones(dim).cuda()

    def forward(self, x, y):
        return super(CosineLoss, self).forward(x, y, self.target)/2


class PearsonLoss(nn.CosineEmbeddingLoss):
    def __init__(self, dim=1):
        super(PearsonLoss, self).__init__()
        self.target = torch.ones(dim).cuda()

    def forward(self, x, y):
        x = x - x.mean()
        y = y - y.mean()
        return super(PearsonLoss, self).forward(x, y, self.target)


class Split2d(nn.Module):
    def __init__(self, kernel_size=(3, 3)):
        super(Split2d, self).__init__()
        self.h, self.w = _pair(kernel_size)
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        output = self.unfold(x).view(x.size(0), x.size(1), self.h, self.w, -1)
        return output.permute(0,4,1,2,3).contiguous().view(-1, x.size(1), self.h, self.w)


class FiveSplit2d(nn.Module):
    def __init__(self, kernel_size):
        super(FiveSplit2d, self).__init__()
        self.split = Split2d(kernel_size)
        self.kernel_size = _pair(kernel_size)

    def forward(self, inputs):
        w, h = self.kernel_size
        x = (inputs.size(-2) - w) // 2
        y = (inputs.size(-1) - h) // 2
        split = self.split(inputs)
        center = inputs[:,:,x:x+w,y:y+h]
        return torch.cat((split, center), dim=0)


class Merge2d(nn.Module):
    def __init__(self, output_size, kernel_size):
        super(Merge2d, self).__init__()
        self.H, self.W = _pair(output_size)
        self.h, self.w = _pair(kernel_size)
        self.fold = nn.Fold(output_size, kernel_size, stride=kernel_size)

    def forward(self, x):
        output = x.view(-1, (self.H//self.h)*(self.W//self.w), x.size(1)*self.h*self.w)
        return self.fold( output.permute(0,2,1).contiguous())


class VerticalFlip(object):
    """Vertically flip the given PIL Image.
    """
    def __init__(self):
        pass

    def __call__(self, img):
        return TF.vflip(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class HorizontalFlip(object):
    """Horizontally flip the given PIL Image.
    """
    def __init__(self):
        pass

    def __call__(self, img):
        return TF.hflip(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomMotionBlur(object):
    def __init__(self, p=[0.7, 0.2, 0.1]):
        self.p = p
        kernel_size = 3
        self.w3 = torch.zeros(4, kernel_size, kernel_size)
        self.w3[0,kernel_size//2,:] = 1.0/kernel_size
        self.w3[1,:,kernel_size//2] = 1.0/kernel_size
        self.w3[2] = torch.eye(kernel_size)
        self.w3[3] = torch.eye(kernel_size).rot90()
        kernel_size = 5
        self.w5 = torch.zeros(4, kernel_size, kernel_size)
        self.w5[0,kernel_size//2,:] = 1.0/kernel_size
        self.w5[1,:,kernel_size//2] = 1.0/kernel_size
        self.w5[2] = torch.eye(kernel_size)
        self.w5[3] = torch.eye(kernel_size).rot90()

    def __call__(self, img):
        """
        Args:
            tensor (Image): Image to be cropped.

        Returns:
            tensor: Random motion blured image.
        """
        p = random.random()
        if p <= self.p[0]:
            return img
        if self.p[0] < p <= self.p[0]+ self.p[1]:
            w = self.w3[torch.randint(0,4,(1,))].unsqueeze(0)
            kernel_size = 3
        elif 1-self.p[2] < p:
            w = self.w5[torch.randint(0,4,(1,))].unsqueeze(0)
            kernel_size = 5

        return F.conv2d(img.unsqueeze(1), w, padding=kernel_size//2).squeeze(1)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class EarlyStopScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                    verbose=False, threshold=1e-4, threshold_mode='rel',
                    cooldown=0, min_lr=0, eps=1e-8):
        super().__init__(optimizer=optimizer, mode=mode, factor=factor, patience=patience,
                            threshold=threshold, threshold_mode=threshold_mode,
                            cooldown=cooldown, min_lr=min_lr, eps=eps, verbose=verbose)
        self.no_decrease = 0

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            return self._reduce_lr(epoch)

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))
                return False
            else:
                return True


class CorrelationSimilarity(nn.Module):
    '''
    Correlation Similarity for multi-channel 2-D tensor(C, H, W) via FFT
    args: input_size: tuple(H, W) --> size of last two dimensions
    Input Shape:
    x: tensor(B, C, H, W)
    y: tensor(N, C, H, W)
    Output Shape:
    o: tensor(B, N)  --> maximum similarity for (x_i, y_j) {i\in [0,B), j\in [0,N)}
    i: tensor(B, N, 2)  --> 2-D translation between x_i and y_j
    '''
    def __init__(self, input_size):
        super(CorrelationSimilarity, self).__init__()
        self.input_size = input_size = _pair(input_size)
        assert(input_size[-1]!=1) # FFT2 is wrong if last dimension is 1
        self.N = math.sqrt(input_size[0]*input_size[1])
        self.fft_args = {'s': input_size, 'dim':[-2,-1], 'norm': 'ortho'}
        self.max = nn.MaxPool2d(kernel_size=input_size)

    def forward(self, x, y):
        X = torch.fft.rfftn(x, **self.fft_args).unsqueeze(1)
        Y = torch.fft.rfftn(y, **self.fft_args)
        g = torch.fft.irfftn((X.conj()*Y).sum(2), **self.fft_args)*self.N
        xx = x.view(x.size(0),-1).norm(dim=-1).view(x.size(0), 1, 1)
        yy = y.view(y.size(0),-1).norm(dim=-1).view(1, y.size(0), 1)
        g = g.view(x.size(0), y.size(0),-1)/xx/yy
        values, indices = torch.max(g, dim=-1)
        indices = torch.stack((indices // self.input_size[1], indices % self.input_size[1]), dim=-1)
        values[values>+1] = +1 # prevent from overflow of  1
        values[values<-1] = -1 # prevent from overflow of -1
        assert((values>+1).sum()==0 and (values<-1).sum()==0)
        return values, indices


class Correlation(nn.Module):
    '''
    Correlation Similarity for multi-channel 2-D patch via FFT
    args: input_size: tuple(H, W) --> size of last two dimensions
    Input Shape:
    x: tensor(B, C, H, W)
    y: tensor(B, C, H, W)
    Output Shape:
    o: tensor(B)
    if accept_translation is False, output is the same with cosine similarity
    '''
    def __init__(self, input_size, accept_translation=True):
        super(Correlation, self).__init__()
        self.accept_translation = accept_translation
        input_size = _pair(input_size)
        assert(input_size[-1]!=1) # FFT2 is wrong if last dimension is 1
        self.N = math.sqrt(input_size[0]*input_size[1])
        self.fft_args = {'s': input_size, 'dim':[-2,-1], 'norm': 'ortho'}
        self.max = nn.MaxPool2d(kernel_size=input_size)

    def forward(self, x, y):
        X = torch.fft.rfftn(x, **self.fft_args)
        Y = torch.fft.rfftn(y, **self.fft_args)
        g = torch.fft.irfftn((X.conj()*Y).sum(2), **self.fft_args)*self.N
        xx = x.view(x.size(0),-1).norm(dim=-1)
        yy = y.view(y.size(0),-1).norm(dim=-1)
        if self.accept_translation is True:
            return self.max(g).view(-1)/xx/yy
        else:
            return g[:,0,0].view(-1)/xx/yy


class CorrelationLoss(Correlation):
    '''
    Correlation Similarity for multi-channel 2-D patch via FFT
    args: input_size: tuple(H, W) --> size of last two dimensions
    Input Shape:
    x: tensor(B, C, H, W)
    y: tensor(B, C, H, W)
    Output Shape:
    o: tensor(1) if 'reduce' is True
    o: tensor(B) if 'reduce' is not True
    '''
    def __init__(self, input_size, reduce = True, accept_translation=True):
        super(CorrelationLoss, self).__init__(input_size, accept_translation)
        self.reduce = reduce

    def forward(self, x, y):
        loss = (1 - super(CorrelationLoss, self).forward(x, y))/2
        if self.reduce is True:
            return loss.mean()
        else:
            return loss


def rolls2d(inputs, shifts, dims=[-2,-1]):
    '''
    shifts: list of tuple/ints for 2-D/1-D roll
    dims: along which dimensions to shift
    inputs: tensor(N, C, H, W); shifts has to be int tensor
    if shifts: tensor(B, N, 2)
        output: tensor(B, N, C, H, W)
    if shifts: tensor(N, 2)
        output: tensor(N, C, H, W)
    '''
    shift_size = shifts.size()
    N, C, H, W = inputs.size()
    assert(shift_size[-1]==2 and N==shift_size[1])
    if len(shift_size) == 2:
        return torch.stack([inputs[i].roll(shifts[i].tolist(), dims) for i in range(N)], dim=0)
    elif len(shift_size) == 3:
        B = shift_size[0]
        o = torch.stack([inputs[i].roll(shifts[j,i].tolist(), dims) for j in range(B) for i in range(N)], dim=0)
        return o.view(B, N, C, H, W)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_batch(batch, name='video', waitkey=1):
    min_v = torch.min(batch)
    range_v = torch.max(batch) - min_v
    if range_v > 0:
        batch = (batch - min_v) / range_v
    else:
        batch = torch.zeros(batch.size())
    grid = torchvision.utils.make_grid(batch, padding=0).cpu()
    img = grid.numpy()[::-1].transpose((1, 2, 0))
    cv2.imshow(name, img)
    cv2.waitKey(waitkey)
    return img


def show_batch_origin(batch, name='video', waitkey=1):
    grid = torchvision.utils.make_grid(batch).cpu()
    img = grid.numpy()[::-1].transpose((1, 2, 0))
    cv2.imshow(name, img)
    cv2.waitKey(waitkey)
    return img
