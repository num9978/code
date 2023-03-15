import torch
from torch import Tensor
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
import math
from typing import Callable, List, Optional, Sequence, Tuple, Union
from torchvision.utils import _log_api_usage_once
import count_ops

def hard_softmax(y_soft, dim=-1):
    # Straight through softmax trick.
    index = y_soft.argmax()
    y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0) # scatter_(int, tensor, (int,tensor))
    return y_hard.detach() - y_soft.detach() + y_soft

class LSQfunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale_factor, scale_factor_grad, n, p):
        ctx.save_for_backward(x, scale_factor)
        ctx.others = scale_factor_grad, n, p
        
        y = x / (scale_factor)
        y = F.hardtanh(y, min_val=n, max_val=p)
        y = torch.round(y)
        y = y * scale_factor

        return y

    @staticmethod
    def backward(ctx, dy):
        x, scale_factor = ctx.saved_tensors
        scale_factor_grad, n, p = ctx.others

        x_sc = x / (scale_factor)

        indicate_small = (x_sc < n).float()
        indicate_big = (x_sc > p).float()
        indicate_mid = 1.0 - indicate_small - indicate_big

        dx = dy * indicate_mid
        dscale_factor = ((indicate_small * n + indicate_big * p + indicate_mid * (torch.round(x_sc) - x_sc)) * dy * scale_factor_grad)

        return dx, dscale_factor, None, None, None

class WLsqQuan(nn.Module):
    def __init__(self, bit, out_planes=None, symmetric=True, per_kernel=True):
        super(WLsqQuan, self).__init__()

        self.bit = bit
        self.out_planes = out_planes
        self.per_kernel = per_kernel

        if self.per_kernel:
            self.s = nn.Parameter(torch.Tensor(self.out_planes))
        else:
            self.s = nn.Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('numel', torch.zeros(1))

    def initialize_lsq(self, x, p):
        # quantized value: [-2**(bit-1)+1, 2**(bit-1)-1]
        if self.per_kernel:
            scale_init = 2.0 * x.detach().view(self.out_planes,-1).abs().mean(dim=1) / math.sqrt(p)
            self.s.data.copy_(scale_init)
        else:
            scale_init = 2.0 * torch.mean(torch.abs(x.detach())) / math.sqrt(p)
            self.s.data.fill_(scale_init)

    def extra_repr(self):
        s = ('bit={bit}, out_planes={out_planes}, per_kernel={per_kernel}')
        return s.format(**self.__dict__)

    def forward(self, x, lin=False):
        ## LSQ
        p = 2 ** (self.bit-1) - 1
        n = -2 ** (self.bit-1) + 1
        if self.training and self.init_state==0:
            self.initialize_lsq(x.data, p)
            self.init_state.fill_(1)
            self.numel.fill_(x.numel())
        sc_grad = 1.0 / math.sqrt(x.numel() * p)
        sc = self.s

        if lin: out = LSQfunction.apply(x, sc.view(-1,1), sc_grad, n, p)
        else: out = LSQfunction.apply(x, sc.view(-1,1,1,1), sc_grad, n, p)
        return out

    def change_bit(self, bit=0):
        self.bit = bit

class ActLsqQuan(nn.Module):
    def __init__(self, bit, tasks, in_planes=None, per_channel=False, symmetric=False, prev_hswish=False):
        super(ActLsqQuan, self).__init__()

        self.bit = bit
        self.tasks = tasks
        self.in_planes = in_planes
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.prev_hswish = prev_hswish
        self.hswish_shift = 0
        if not symmetric and prev_hswish:
            self.hswish_shift = 3/8

        for i, t in enumerate(tasks):
            setattr(self, f's_{i}', nn.Parameter(torch.Tensor(1)))
        self.alpha = nn.Parameter(torch.randn(len(tasks))*1e-3)
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('numel', torch.zeros(1))

    def initialize_lsq(self, x):
        for i, bit in enumerate(self.tasks):
            n_lv = 2 ** bit - 1
            if self.per_channel:
                scale_init = 2.0 * x.detach().permute(1,0,2,3).contiguous().view(self.in_planes,-1).abs().mean(dim=1) / math.sqrt(n_lv)
                getattr(self, f's_{i}').data.copy_(scale_init)
            else:
                scale_init = 2.0 * torch.mean(torch.abs(x)) / math.sqrt(n_lv)
                getattr(self, f's_{i}').data.fill_(scale_init)

    def extra_repr(self):
        s = ('bit={bit}, tasks={tasks}, in_planes={in_planes}, per_channel={per_channel}, symmetric={symmetric}, prev_hswish={prev_hswish}')
        return s.format(**self.__dict__)

    def forward(self, x):
        ## LSQ Bit-meta
        if not isinstance(self.bit, list):
            if self.symmetric:
                p = 2 ** (self.bit-1) - 1
                n = -2 ** (self.bit-1) + 1
            else:
                p = 2 ** self.bit - 1
                n = 0
            if self.training and self.init_state==0:
                self.initialize_lsq(x.data)
                self.init_state.fill_(1)
                self.numel.fill_(x.numel())
            idx = self.tasks.index(self.bit)
            sc_grad = 1.0 / math.sqrt(x.numel() * p)
            sc = getattr(self, f's_{idx}')

            x = x + self.hswish_shift
            out = LSQfunction.apply(x, sc, sc_grad, n, p)
            out = out - self.hswish_shift
            return out
        ## LSQ Bit-search
        else:
            if self.symmetric:
                ps = [2**(b-1)-1 for b in self.bit]
                ns = [-2**(b-1)+1 for b in self.bit]
            else:
                ps = [2**b-1 for b in self.bit]
                ns = [0 for _ in self.bit]
            sc_grads = [1.0 / math.sqrt(x.numel() * n_lv) for n_lv in ps]
            scs = [getattr(self, f's_{idx}') for idx in range(len(ps))]

            alphas = F.softmax(self.alpha)
            x = x + self.hswish_shift
            if self.per_channel:
                qouts = [LSQfunction.apply(x, sc.view(1,-1,1,1), sc_grad, n, p) for sc, sc_grad, n, p in zip(scs, sc_grads, ns, ps)]
            else:
                qouts = [LSQfunction.apply(x, sc, sc_grad, n, p) for sc, sc_grad, n, p in zip(scs, sc_grads, ns, ps)]
            for i in range(len(qouts)):
                qouts[i] = qouts[i] - self.hswish_shift
            out = sum([alpha*qout for alpha, qout in zip(alphas, qouts)])
            return out

    def change_precision(self, a_bin=False):
        self.a_bin = a_bin

    def change_bit(self, bit=0):
        self.bit = bit

    def set_searched_bit(self):
        alpha = F.softmax(self.alpha)
        bit_idx = alpha.argmax().item()
        self.bit = self.tasks[bit_idx]
        print(f"{alpha} | {F.softmax(self.alpha)} | {self.bit}-bit")

class FixedActLsqQuan(nn.Module):
    def __init__(self, bit, in_planes=None, per_channel=False, symmetric=False, prev_hswish=False):
        super(FixedActLsqQuan, self).__init__()

        self.bit = bit
        self.in_planes = in_planes
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.prev_hswish = prev_hswish
        self.hswish_shift = 0
        if not symmetric and prev_hswish:
            self.hswish_shift = 3/8
        
        self.scale =  nn.Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('numel', torch.zeros(1))

    def initialize_lsq(self, x, p):
        scale_init = 2.0 * torch.mean(torch.abs(x)) / math.sqrt(p)
        self.scale.data.fill_(scale_init)

    def extra_repr(self):
        s = ('bit={bit}, in_planes={in_planes}, per_channel={per_channel}, symmetric={symmetric}, prev_hswish={prev_hswish}')
        return s.format(**self.__dict__)

    def forward(self, x):
        if self.symmetric:
            p = 2 ** (self.bit-1) - 1
            n = -2 ** (self.bit-1) + 1
        else:
            p = 2 ** self.bit - 1
            n = 0
        if self.training and self.init_state==0:
            self.initialize_lsq(x.data, p)
            self.init_state.fill_(1)
            self.numel.fill_(x.numel())
        sc_grad = 1.0 / math.sqrt(x.numel() * p)
        sc = self.scale

        x = x + self.hswish_shift
        if self.per_channel:
            out = LSQfunction.apply(x, sc.view(1,-1,1,1), sc_grad, n, p)
        else:
            out = LSQfunction.apply(x, sc, sc_grad, n, p)
        out = out - self.hswish_shift
        return out

    def change_bit(self, bit=0):
        self.bit = bit
        
class QConv_fixed(nn.Conv2d):
    def __init__(self, *args, **kargs):
        symmetric = kargs.pop('symmetric')
        prev_hswish = kargs.pop('prev_hswish')
        tasks = kargs.pop('tasks')
        super(QConv_fixed, self).__init__(*args, **kargs)
        self.symmetric = symmetric
        self.prev_hswish = prev_hswish
        self.a_bin = True
        self.w_bin = False
        self.a_quantize = FixedActLsqQuan(8, in_planes=self.in_channels, per_channel=False, symmetric=symmetric, prev_hswish=prev_hswish)
        self.w_quantize = WLsqQuan(8, out_planes=self.out_channels)

        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('ops', torch.zeros(1))

    def forward(self, x):
        if self.a_bin: qa = self.a_quantize(x)
        else: qa = x
        if self.w_bin: qw = self.w_quantize(self.weight)
        else: qw = self.weight

        y = F.conv2d(qa, qw, self.bias, self.stride, self.padding, self.dilation, self.groups)

        if self.training and self.init_state==0:
            self.ops.fill_(count_ops._count_convNd(self, y))
            self.init_state.fill_(1)

        return y

    def change_precision(self, a_bin=True, w_bin=False):
        self.a_bin = a_bin
        self.w_bin = w_bin

class QConv(nn.Conv2d):
    def __init__(self, *args, **kargs):
        symmetric = kargs.pop('symmetric')
        prev_hswish = kargs.pop('prev_hswish')
        tasks = kargs.pop('tasks')
        super(QConv, self).__init__(*args, **kargs)
        self.symmetric = symmetric
        self.prev_hswish = prev_hswish
        self.tasks = tasks
        self.a_bin = True
        self.a_quantize = ActLsqQuan(4, tasks, in_planes=self.in_channels, per_channel=False, symmetric=symmetric, prev_hswish=prev_hswish)
        self.w_bin = False
        self.w_quantize = WLsqQuan(4, out_planes=self.out_channels)

        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('ops', torch.zeros(1))

    def forward(self, x):
        if self.a_bin: qa = self.a_quantize(x)
        else: qa = x
        if self.w_bin: qw = self.w_quantize(self.weight)
        else: qw = self.weight

        y = F.conv2d(qa, qw, self.bias, self.stride, self.padding, self.dilation, self.groups)

        if self.training and self.init_state==0:
            self.ops.fill_(count_ops._count_convNd(self, y))
            self.init_state.fill_(1)

        return y

    def change_precision(self, a_bin=True, w_bin=False):
        self.a_bin = a_bin
        self.w_bin = w_bin

class QLinear(nn.Linear):
    def __init__(self, *args, **kargs):
        tasks = kargs.pop('tasks')
        super(QLinear, self).__init__(*args, **kargs)
        self.tasks = tasks
        self.a_bin = True
        self.a_quantize = ActLsqQuan(4, tasks, in_planes=self.in_features, per_channel=False, symmetric=False, prev_hswish=True)
        self.w_bin = False
        self.w_quantize = WLsqQuan(8, out_planes=self.out_features)

        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('ops', torch.zeros(1))

    def forward(self, x):
        if self.a_bin: qa = self.a_quantize(x)
        else: qa = x
        if self.w_bin: qw = self.w_quantize(self.weight, lin=True)
        else: qw = self.weight

        y = F.linear(qa, qw, self.bias)

        if self.training and self.init_state==0:
            self.ops.fill_(count_ops._count_linear(self, y))
            self.init_state.fill_(1)

        return y

    def change_precision(self, a_bin=True, w_bin=False):
        self.a_bin = a_bin
        self.w_bin = w_bin

class QLinear_fixed(nn.Linear):
    def __init__(self, *args, **kargs):
        super(QLinear_fixed, self).__init__(*args, **kargs)
        self.a_bin = True
        self.a_quantize = FixedActLsqQuan(8, in_planes=self.in_features, per_channel=False, symmetric=False, prev_hswish=True)
        self.w_bin = False
        self.w_quantize = WLsqQuan(8, out_planes=self.out_features)

        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('ops', torch.zeros(1))

    def forward(self, x):
        if self.a_bin: qa = self.a_quantize(x)
        else: qa = x
        if self.w_bin: qw = self.w_quantize(self.weight, lin=True)
        else: qw = self.weight

        y = F.linear(qa, qw, self.bias)

        if self.training and self.init_state==0:
            self.ops.fill_(count_ops._count_linear(self, y))
            self.init_state.fill_(1)

        return y

    def change_precision(self, a_bin=True, w_bin=False):
        self.a_bin = a_bin
        self.w_bin = w_bin

class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, ...]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
        symmetric: bool = False,
        prev_hswish: bool = False,
        tasks: list = [],
    ) -> None:

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                symmetric=symmetric,
                prev_hswish=prev_hswish,
                tasks=tasks,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        _log_api_usage_once(self)
        self.out_channels = out_channels

        if self.__class__ == ConvNormActivation:
            warnings.warn(
                "Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead."
            )

class Conv2dNormActivation(ConvNormActivation):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = QConv,
        symmetric: bool = False,
        prev_hswish: bool = False,
        tasks: list = [],
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            conv_layer,
            symmetric,
            prev_hswish,
            tasks,
        )
        
class SqueezeExcitation(torch.nn.Module):

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
        prev_hswish: bool = False,
        tasks: list = [],
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        # self.fc1 = QConv_fixed(input_channels, squeeze_channels, 1, symmetric=False, tasks=tasks, prev_hswish=prev_hswish)
        self.fc1 = QConv(input_channels, squeeze_channels, 1, symmetric=False, tasks=tasks, prev_hswish=prev_hswish)
        # self.fc2 = QConv_fixed(squeeze_channels, input_channels, 1, symmetric=False, tasks=tasks, prev_hswish=True if isinstance(activation, nn.Hardswish) else False)
        self.fc2 = QConv(squeeze_channels, input_channels, 1, symmetric=False, tasks=tasks, prev_hswish=True if isinstance(activation, nn.Hardswish) else False)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input
