#!/bin/python3
# A progress bar which can print the progress
# modified from https://github.com/xinntao/BasicSR/blob/master/codes/scripts/extract_subimgs_single.py
import os
import sys
import cv2
import shutil
import numpy as np
import argparse
import h5py
import random
import scipy.io as sio
from multiprocessing import Pool
from utils.progress_bar import ProgressBar


def main():
    parser = argparse.ArgumentParser('Extract sub images from Hyperspectral images.')
    parser.add_argument('--path1', type=str, default='E:/HSI/CAVE/CAVE/')
    parser.add_argument('--path2', type=str, default='E:/HSI/Harvard/Harvard/')
    parser.add_argument('--path3', type=str, default='E:/HSI/NTIRE2018/NTIRE2018_Train1_Spectral/')
    parser.add_argument('-o', '--out-path', type=str, default='E:/HSI/HSI_MIX/')
    parser.add_argument('-t', '--threads', type=int, default=1)
    parser.add_argument('-c', '--crop-size', type=int, default=256)
    parser.add_argument('-s', '--stride', type=int, default=96)

    opt = parser.parse_args()
    in_path1 = opt.path1
    in_path2 = opt.path2
    in_path3 = opt.path3
    out_path = opt.out_path
    n_threads = opt.threads
    crop_sz = opt.crop_size
    stride = opt.stride
    thres_sz = 30  #
    compression_level = 3

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    img_list = []
    for root, _, file_list in sorted(os.walk(in_path1)):
        path = [os.path.join(root, x) for x in file_list]
        img_list.extend(path)
    for root, _, file_list in sorted(os.walk(in_path2)):
        path = [os.path.join(root, x) for x in file_list]
        img_list.extend(path)
    for root, _, file_list in sorted(os.walk(in_path3)):
        path = [os.path.join(root, x) for x in file_list]
        img_list.extend(path)

    l = len(img_list)
    train_ids = list(set([random.randrange(l) for _ in range(l)]))
    train_ids.sort()
    val_ids = [idx for idx in range(l) if idx not in train_ids]
    val_ids.sort()
    train_files = [img_list[idx] for idx in train_ids]
    val_files = [img_list[idx] for idx in val_ids]
    sio.savemat(os.path.join(out_path, 'filenames.mat'),
                {'train_files': train_files,
                 'val_files': val_files})

    def update(arg):
        pbar.update(arg)

    pbar = ProgressBar(len(train_files))

    pool = Pool(n_threads)
    for path in train_files:
        pool.apply_async(
            worker,
            args=(path, os.path.join(out_path, 'train.h5'), crop_sz, stride, thres_sz, compression_level),
            callback=update
        )
    pool.close()
    pool.join()
    print("-----------Generation Finish-------------")

    val_path = os.path.join(out_path, 'val')
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    for path in val_files:
        shutil.copy2(path, val_path)
    print("-----------Copy Val-files Finish-------------")


def worker(path, save_folder, crop_sz, stride, thres_sz, compression_level):
    img_name = os.path.basename(path)
    f = h5py.File(path, 'r')
    img = np.array(f['rad'])
    # img = img.swapaxes(axis1=0, axis2=2)
    f.close()

    print("img_name [{:s}]\t maxValue [{:.6f}]".format(img_name, np.max(img)))
    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        c, h, w = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))

    h_space = np.arange(0, h - crop_sz + 1, stride)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, stride)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[:, x:x + crop_sz, y:y + crop_sz]
                crop_img = np.ascontiguousarray(crop_img)
                crop_img = crop_img.swapaxes(0, 2)
            with h5py.File(save_folder) as f:
                f.create_dataset(img_name.replace('.mat', '_s{:03d}'.format(index)), data=crop_img, compression=compression_level)
    return 'Processing {:s} ...'.format(img_name)


if __name__ == '__main__':
    main()
