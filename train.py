import os
import sys
import time
import math
import random
import torch
import argparse
import options.option as option
from utils import util
from models import create_model


def main():
    parser = argparse.ArgumentParser("DP_HSISR")
    parser.add_argument('-opt', type=str, default='options/train/train_DP_HSISR.json')
    opt = option.parse(parser.parse_args().opt, is_train=True)


if __name__ == '__main__':
    main()
