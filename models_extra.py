# -*- coding: utf-8 -*-
"""Additional B1 baseline: PDNO (pseudo-differential neural operator, continuous symbol).
The symbol(kx,ky,a,s) is a continuous MLP of the wavenumbers (integer Fourier modes, independent of resolution) and the conditions;
it acts channel-wise over all rfft2 wavenumbers, hence discretization invariant and evaluable at any resolution; channels are mixed with 1x1 convolution.
It gives a learned-symbol-vs-analytic-symbol contrast to FrFNO (analytic fixed symbol plus learned residual)."""
import torch, torch.nn as nn
import torch.nn.functional as F


class PDIO(nn.Module):
    def __init__(self, c, hid=128):
        super().__init__(); self.c = c
        self.mlp = nn.Sequential(nn.Linear(5, hid), nn.GELU(), nn.Linear(hid, hid),
                                 nn.GELU(), nn.Linear(hid, 2 * c))
        self.conv1 = nn.Conv2d(c, c, 1)
        self.cache = {}

    def _kfeat(self, H, W, dev, dt):
        if (H, W) not in self.cache:
            kx = torch.fft.fftfreq(H, d=1. / H, device=dev, dtype=dt)          # integer modes (independent of resolution)
            ky = torch.fft.rfftfreq(W, d=1. / W, device=dev, dtype=dt)
            KX, KY = torch.meshgrid(kx, ky, indexing='ij'); K = torch.sqrt(KX ** 2 + KY ** 2)
            feat = torch.stack([torch.log1p(K), KX / (K + 1e-6), KY / (K + 1e-6)], -1)  # (H,Wf,3)
            self.cache[(H, W)] = feat
        return self.cache[(H, W)]

    def forward(self, x, ach):
        B, c, H, W = x.shape; Wf = W // 2 + 1
        kf = self._kfeat(H, W, x.device, x.dtype)                            # (H,Wf,3)
        cond = ach.view(B, 1, 1, 2).expand(B, H, Wf, 2)
        z = torch.cat([kf[None].expand(B, H, Wf, 3), cond], -1)              # (B,H,Wf,5)
        g = self.mlp(z).reshape(B, H, Wf, c, 2)
        gc = torch.view_as_complex(g.contiguous())                           # (B,H,Wf,c)
        xf = torch.fft.rfft2(x).permute(0, 2, 3, 1)                          # (B,H,Wf,c)
        out = torch.fft.irfft2((xf * gc).permute(0, 3, 1, 2), s=(H, W))
        return out + self.conv1(x)


class PDNO2d(nn.Module):
    def __init__(self, in_dim=2, width=72, n_layers=6, hid=160, emb=32, seed=6):
        super().__init__(); torch.manual_seed(seed)
        self.act = nn.GELU(); self.nl = n_layers
        self.fc0 = nn.Sequential(nn.Linear(in_dim + 4, 128), self.act, nn.Linear(128, width))
        self.pd = nn.ModuleList([PDIO(width, hid) for _ in range(n_layers)])
        self.enc = nn.Sequential(nn.Linear(2, emb), nn.GELU(), nn.Linear(emb, emb))
        self.fg = nn.ModuleList([nn.Linear(emb, width) for _ in range(n_layers)])
        self.fb = nn.ModuleList([nn.Linear(emb, width) for _ in range(n_layers)])
        self.q = nn.Sequential(nn.Linear(width, 128), self.act, nn.Linear(128, 1))
        self.to('cuda')

    def forward(self, x, cond):
        a, s = float(cond[0]), float(cond[1]); B, H, W, _ = x.shape
        gx = torch.linspace(0, 1, W, device=x.device).view(1, 1, W, 1).repeat(B, H, 1, 1)
        gy = torch.linspace(0, 1, H, device=x.device).view(1, H, 1, 1).repeat(B, 1, W, 1)
        ach = torch.tensor([[a, s]], device=x.device, dtype=x.dtype).repeat(B, 1)
        ac = ach.view(B, 1, 1, 2).expand(B, H, W, 2)
        h = self.fc0(torch.cat([gy, gx, x, ac], -1)).permute(0, 3, 1, 2)
        e = self.enc(ach)
        for i in range(self.nl):
            h = self.pd[i](h, ach)
            gg = (1 + self.fg[i](e)).unsqueeze(-1).unsqueeze(-1); bb = self.fb[i](e).unsqueeze(-1).unsqueeze(-1)
            h = gg * h + bb
            if i != self.nl - 1: h = self.act(h)
        return self.q(h.permute(0, 2, 3, 1)).squeeze(-1)
