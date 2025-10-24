#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在不依赖 timm 库的情况下，构造与 test_time.py 中相同的 Stereo 实例。

做法：预先注册一个简化版的 timm 模块，只实现 create_model 所需的接口，
返回一个轻量级的 MobilenetV2 风格骨干网络（层结构保持与配置一致，
但权重为随机初始化）。随后复用 test_time.py 里的 load_configs 与 Stereo。
"""

import sys
import types
from pathlib import Path

import torch
import torch.nn as nn



class MBConv(nn.Module):
    """极简版的 MBConv block, 用于维持通道数与下采样结构。"""

    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()
        mid = max(in_ch, out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class MobileNetV2Stub(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv_stem = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        blocks = [
            MBConv(32, 16, stride=1),
            MBConv(16, 24, stride=2),
            MBConv(24, 24, stride=1),
            MBConv(24, 32, stride=2),
            MBConv(32, 96, stride=2),
            MBConv(96, 160, stride=1),
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - 未在此脚本中使用
        x = self.conv_stem(x) #一般的conv2d Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        x = self.bn1(x) # BN(32) + ReLU6
        x2 = self.block0(x)
        x4 = self.block1(x2)

        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)

        x_out = [x4, x8, x16, x32]

        return x2, x_out


def build_stereo(config_path: Path, dataname: str = 'analysis'):
    """加载配置并返回与 test_time.py 中一致的 Stereo 实例。"""
    _install_timm_stub()
    from test_time import Stereo, load_configs

    cfg = load_configs(str(config_path))
    return Stereo(cfg, dataname=dataname)


def main() -> None:
    config = Path('configs/stereo/cfg_coex.yaml')
    stereo = build_stereo(config)
    inner = stereo.stereo

    print('Stereo wrapper type:', type(stereo))
    print('Inner stereo type:', type(inner))
    print('Backbone type:', stereo.cfg['model']['stereo']['backbone']['type'])


if __name__ == '__main__':
    main()
