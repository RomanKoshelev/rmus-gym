#!/usr/bin/env python
from alg.PeterKovacs.ddgp import DDGP
from launch import launch

if __name__ == '__main__':
    launch('train', 'unit', DDGP, 'Tentacle-v0')
