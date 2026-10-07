# -*- coding: utf-8 -*-
"""
Unified model library for surrogate model comparison (forward problem, surrogate modeling)。Unified interface: forward(x, cond)
  x   : (B, C, H, W) spatial input channels [u0, nu] (C=2)
  cond: (B, 2) = (alpha, s)
  returns : (B, H, W) = prediction u(T)
- CNOWrap   : ready-to-use CNO2d_simplified (pure PyTorch, size bound at construction)
- DeepONet2d: branch (fixed-sensor encoded input field) + trunk (coordinates+conditions), output resolution follows input
- UNet2d    : fully convolutional compact U-Net, final interpolation back to input size (structurally runs at any resolution, but no discretization-invariance guarantee)
Standard FNO and FrFNO reuse the same channels-last DualNet defined in frfno_core.py.
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local vendored copy of CNO2d_simplified (resolved relative to this file,
# so the project no longer depends on any absolute non-ASCII path).
CNO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CNO2d_simplified')
if CNO_PATH not in sys.path:
    sys.path.insert(0, CNO_PATH)
from CNO2d import CNO2d

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DT = torch.float32

# ---------------- CNO ----------------
class CNOWrap(nn.Module):
    def __init__(self, size=17, in_c=2, ch=16, n_layers=3, n_res=4, seed=0, use_bn=True):
        super().__init__(); torch.manual_seed(seed); self.size = size
        self.cno = CNO2d(in_dim=in_c+2, out_dim=1, size=size, N_layers=n_layers,
                         N_res=n_res, N_res_neck=n_res, channel_multiplier=ch, use_bn=use_bn)
    def forward(self, x, cond):
        B, C, H, W = x.shape
        cc = cond.view(B, 2, 1, 1).expand(B, 2, H, W)
        z = torch.cat([x, cc], 1)
        if H != self.size:   # CNO size bound at construction: at non-training resolution, coarsen to size, forward pass, then interpolate back
            z = F.interpolate(z, size=(self.size, self.size), mode='bicubic', antialias=True)
            y = self.cno(z)
            return F.interpolate(y, size=(H, W), mode='bicubic', antialias=True).squeeze(1)
        return self.cno(z).squeeze(1)

# ---------------- DeepONet ----------------
class BranchCNN(nn.Module):
    def __init__(self, in_c, p, w=128):
        super().__init__()
        # sensor=17 -> AvgPool twice to 4x4, preserving spatial structure (avoid global pooling that flattens input field morphology)
        self.net = nn.Sequential(
            nn.Conv2d(in_c, w//2, 3, 1, 1), nn.LeakyReLU(),
            nn.AvgPool2d(2),                                   # 17->8
            nn.Conv2d(w//2, w, 3, 1, 1), nn.LeakyReLU(),
            nn.AvgPool2d(2),                                   # 8->4
            nn.Conv2d(w, w, 3, 1, 1), nn.LeakyReLU(),          # 4x4
            nn.Flatten(),
            nn.Linear(w*4*4, w), nn.LeakyReLU(), nn.Linear(w, p))
    def forward(self, x): return self.net(x)

class DeepONet2d(nn.Module):
    def __init__(self, in_c=2, p=128, sensor=17, seed=0, w=256):
        super().__init__(); torch.manual_seed(seed); self.p = p; self.sensor = sensor
        self.branch = BranchCNN(in_c+2, p, w)
        self.trunk = nn.Sequential(nn.Linear(4, w), nn.LayerNorm(w), nn.LeakyReLU(),
                                   nn.Linear(w, w), nn.LayerNorm(w), nn.LeakyReLU(),
                                   nn.Linear(w, p))
    def forward(self, x, cond):
        B, C, H, W = x.shape
        xs = F.interpolate(x, size=(self.sensor, self.sensor), mode='bilinear', align_corners=False)
        cc = cond.view(B, 2, 1, 1).expand(B, 2, self.sensor, self.sensor)
        w = self.branch(torch.cat([xs, cc], 1))               # (B,p)
        gx = torch.linspace(0, 1, W, device=x.device); gy = torch.linspace(0, 1, H, device=x.device)
        GX, GY = torch.meshgrid(gy, gx, indexing='ij')
        coord = torch.stack([GX, GY], -1).reshape(1, H*W, 2).expand(B, H*W, 2)          # (B,Q,2)
        cb = cond.view(B, 1, 2).expand(B, H*W, 2)
        t = self.trunk(torch.cat([coord, cb], -1))                                       # (B,Q,p)
        out = torch.einsum('bp,bqp->bq', w, t) / self.p**0.5
        return out.reshape(B, H, W)

# ---------------- UNet (pure-convolution baseline) ----------------
class DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ci, co, 3, 1, 1), nn.LeakyReLU(),
                                 nn.Conv2d(co, co, 3, 1, 1), nn.LeakyReLU())
    def forward(self, x): return self.net(x)

class UNet2d(nn.Module):
    def __init__(self, in_c=2, base=32, seed=0):
        super().__init__(); torch.manual_seed(seed)
        self.d1 = DoubleConv(in_c+2, base)
        self.d2 = DoubleConv(base, base*2)
        self.d3 = DoubleConv(base*2, base*4)
        self.bot = DoubleConv(base*4, base*4)
        self.u3 = DoubleConv(base*8, base*2)
        self.u2 = DoubleConv(base*4, base)
        self.u1 = nn.Conv2d(base*2, 1, 1)
        self.pool = nn.AvgPool2d(2)
    def forward(self, x, cond):
        B, C, H, W = x.shape
        cc = cond.view(B, 2, 1, 1).expand(B, 2, H, W); z = torch.cat([x, cc], 1)
        x1 = self.d1(z)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))
        y = self.bot(self.pool(x3))
        y = F.interpolate(y, size=x3.shape[-2:], mode='bilinear', align_corners=False)
        y = self.u3(torch.cat([y, x3], 1))
        y = F.interpolate(y, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        y = self.u2(torch.cat([y, x2], 1))
        y = F.interpolate(y, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        y = self.u1(torch.cat([y, x1], 1))
        if y.shape[-1] != W or y.shape[-2] != H:   # restore the input shape
            y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
        return y.squeeze(1)

def n_params(net): return sum(p.numel() for p in net.parameters())

if __name__ == '__main__':
    # quick shape / parameter-count / cross-resolution self-check
    for name, net in [('CNO', CNOWrap(17)), ('DeepONet', DeepONet2d()), ('UNet', UNet2d())]:
        net = net.to(DEV).float()
        x = torch.randn(2, 2, 17, 17, device=DEV); c = torch.rand(2, 2, device=DEV)
        y17 = net(x, c)
        xb = torch.randn(2, 2, 65, 65, device=DEV)
        y65 = net(xb, c)
        print('%-8s params=%.1fK  out17=%s  out65=%s' %
              (name, n_params(net)/1e3, tuple(y17.shape), tuple(y65.shape)))
