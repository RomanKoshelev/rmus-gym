#!/usr/bin/env python
from alg.RomanKoshelev.sa_wrapper import SAWrapper
from launch import launch

if __name__ == '__main__':
    launch('train', 'superagent', SAWrapper, 'Tentacle-v0')
