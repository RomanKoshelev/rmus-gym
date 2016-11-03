#!/usr/bin/env python
from alg.PeterKovacs.ddpg import DDPG
from launch import launch

if __name__ == '__main__':
    launch('run', 'unit', DDPG, 'Tentacle-v0', steps=100)
