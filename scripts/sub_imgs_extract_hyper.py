#!/bin/python3
# A progress bar which can print the progress
# modified from https://github.com/xinntao/BasicSR/blob/master/codes/scripts/extract_subimgs_single.py
import os
import sys
import cv2
import numpy as np
import argparse
import h5py
import scipy.io as sio
from multiprocessing import Pool
from utils.progress_bar import ProgressBar


def main():
    parser = argparse.ArgumentParser('Extract sub images from Hyperspectral images.')
    parser.add_argument('-p', '--path', type=str, default='E:/HSI/NTIRE2018/NTIRE2018_Train1_Spectral/')
    parser.add_argument('-o', '--out-path', type=str, default='')
    parser.add_argument('-t', '--threads', type=int, default=8)
    parser.add_argument('-c', '--crop-size', type=int, default=256)
    parser.add_argument('-s', '--stride', type=int, default=96)

    opt = parser.parse_args()
    in_path = opt.path
    out_path = opt.out_path
    n_threads = opt.threads
    crop_sz = opt.crop_size
    stride = opt.stride
    thres_sz = 30  #
    if out_path == '':
        dataset_name = in_path.split('/')[-2].split('\\')[-1]
        out_path = in_path.replace(dataset_name, dataset_name + '_sub')

    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print('mkdir [{:s}] ...'.format(out_path))
    else:
        print('[*] Folder [{:s}] already exists.'.format(out_path))
        # return

    img_list = []
    for root, _, file_list in sorted(os.walk(in_path)):
        path = [os.path.join(root, x) for x in file_list]
        img_list.extend(path)

    def update(arg):
        pbar.update(arg)

    pbar = ProgressBar(len(img_list))

    pool = Pool(n_threads)
    for path in img_list:
        pool.apply_async(
            worker,
            args=(path, out_path, crop_sz, stride, thres_sz),
            callback=update
        )
    pool.close()
    pool.join()
    print("-----------Generation Finish-------------")


def worker(path, save_folder, crop_sz, stride, thres_sz):
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
            sio.savemat(
                os.path.join(save_folder, img_name.replace('.mat', '_s{:03d}.mat'.format(index))),
                {'img': crop_img}
            )
    return 'Processing {:s} ...'.format(img_name)


if __name__ == '__main__':
    main()
