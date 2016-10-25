#!/usr/bin/env python
from alg.PeterKovacs.ddqn import DDQN
from launch import launch

if __name__ == '__main__':
    launch('train', 'unit', DDQN, 'Tentacle-v0')
