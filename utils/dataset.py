import os
import math
import random
import numpy as np
import h5py
import torch
import torch.utils.data as data
from utils.util import calculate_weights_indices


class HSIDataset(data.dataset):
    """
    Read HR HSIs `Z` and generate LR HSIs `H` and conventional images `M`
    """
    def __init__(self, opt):
        super(HSIDataset, self).__init__()
        self.opt = opt  # opt['datasets']
        self.path = self.opt['path']
        self.keys = self._get_paths_from_h5(self.path)
        self.data_mode = opt['data_mode']  # HWC or CHW

    def __getitem__(self, index):
        HR_HSI = self._read_image_from_h5(self.path, self.keys[index])
        LR_HSI = self._resize_HSI(HR_HSI, self.opt['scale'])
        HR_MSI = self._spectral_convert(HR_HSI, self.opt['R_mode'])
        return {'HR_HSI': HR_HSI, 'LR_HSI': LR_HSI, 'HR_MSI': HR_MSI}

    def __len__(self):
        return len(self.keys)

    def _get_keys_from_h5(self, path):
        assert os.path.isfile(path) and path.endswith('.h5'), "`{:s}` is not a valid file.".format(path)
        with h5py.File(path) as f:
            keys = f.keys()
        return list(keys)

    def _read_image_from_h5(self, path, key):
        assert os.path.isfile(path) and path.endswith('.h5'), "`{:s}` is not a valid file.".format(path)
        with h5py.File(path) as f:
            image = f[key]
        return image

    def _resize_HSI(self, HR, scale, antialiasing=True):
        """
        HSI imresize method
        modified from mhttps://github.com/xinntao/BasicSR/blob/master/codes/data/util.py
        """
        if HR.ndim != 3:
            raise ValueError("Input HR must have ndim == 3 but found ndim = {:s}".format(HR.ndim))
        in_H, in_W, in_C = HR.shape
        h_dim, w_dim, c_dim = 0, 1, 2
        if self.data_mode is None or self.data_mode == '':
            self.data_mode = 'HWC' if in_C < in_H else 'CHW'
        if self.data_mode == 'CHW':
            in_H, in_W, in_C = in_W, in_C, in_H
            h_dim, w_dim, c_dim = 1, 2, 0
        out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
        kernel_width = 4
        kernel = 'cubic'

        img = torch.from_numpy(HR)

        # Return the desired dimension order for performing the resize.  The
        # strategy is to perform the resize first along the dimension with the
        # smallest scale factor.
        # Now we do not support this.

        # get weights and indices
        weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
            in_H, out_H, scale, kernel, kernel_width, antialiasing)
        weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
            in_W, out_W, scale, kernel, kernel_width, antialiasing)
        # process H dimension
        # symmetric copying
        if c_dim == 2:
            img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
        else:
            img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
        img_aug.narrow(h_dim, sym_len_Hs, in_H).copy_(img)

        sym_patch = img[:sym_len_Hs, :, :] if c_dim == 2 else img[:, :sym_len_Hs, :]
        inv_idx = torch.arange(sym_patch.size(h_dim) - 1, -1, -1).long()
        sym_patch_inv = sym_patch.index_select(h_dim, inv_idx)
        img_aug.narrow(h_dim, 0, sym_len_Hs).copy_(sym_patch_inv)

        sym_patch = img[-sym_len_He:, :, :] if c_dim == 2 else img[:, -sym_len_Hs:, :]
        inv_idx = torch.arange(sym_patch.size(h_dim) - 1, -1, -1).long()
        sym_patch_inv = sym_patch.index_select(h_dim, inv_idx)
        img_aug.narrow(h_dim, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

        out_1 = torch.FloatTensor(out_H, in_W, in_C) if c_dim == 2 else torch.FloatTensor(in_C, out_H, in_W)
        kernel_width = weights_H.size(1)
        for i in range(out_H):
            idx = int(indices_H[i][0])
            if c_dim == 2:
                for c in range(in_C):
                    out_1[i, :, c] = img_aug[idx:idx + kernel_width, :, c].transpose(0, 1).mv(weights_H[i])
            else:
                for c in range(in_C):
                    out_1[c, i, :] = img_aug[c, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])

        # process W dimension
        # symmetric copying
        out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C) \
                if c_dim == 2 else torch.FloatTensor(in_C, out_H, in_W + sym_len_Ws + sym_len_We)
        out_1_aug.narrow(w_dim, sym_len_Ws, in_W).copy_(out_1)

        sym_patch = out_1[:, :sym_len_Ws, :]
        inv_idx = torch.arange(sym_patch.size(w_dim) - 1, -1, -1).long()
        sym_patch_inv = sym_patch.index_select(w_dim, inv_idx)
        out_1_aug.narrow(w_dim, 0, sym_len_Ws).copy_(sym_patch_inv)

        sym_patch = out_1[:, -sym_len_We:, :]
        inv_idx = torch.arange(sym_patch.size(w_dim) - 1, -1, -1).long()
        sym_patch_inv = sym_patch.index_select(w_dim, inv_idx)
        out_1_aug.narrow(w_dim, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

        out_2 = torch.FloatTensor(out_H, out_W, in_C) if c_dim == 2 else torch.FloatTensor(in_C, out_H, in_W)
        kernel_width = weights_W.size(1)
        for i in range(out_W):
            idx = int(indices_W[i][0])
            if c_dim == 2:
                for c in range(in_C):
                    out_2[:, i, c] = out_1_aug[:, idx:idx + kernel_width, c].mv(weights_W[i])
            else:
                for c in range(in_C):
                    out_2[c, :, i] = out_1_aug[c, :, idx:idx + kernel_width].mv(weights_W[i])

        return out_2.numpy()

    def _spectral_convert(self, HSI, R_mode='D700'):
        if R_mode == 'D700':
            R = torch.from_numpy(np.array(
                [[0.005, 0.007, 0.012, 0.015, 0.023, 0.025, 0.030, 0.026, 0.024, 0.019, 0.010, 0.004, 0.000, 0.000,
                  0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                  0.000, 0.000, 0.000],
                 [0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.002, 0.003, 0.005, 0.007, 0.012, 0.013, 0.015, 0.016,
                  0.017, 0.020, 0.013, 0.011, 0.009, 0.005, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
                  0.002, 0.002, 0.003],
                 [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                  0.000, 0.000, 0.001, 0.003, 0.010, 0.012, 0.013, 0.022, 0.020, 0.020, 0.018, 0.017, 0.016, 0.016,
                  0.014, 0.014, 0.013]]))
        else:
            raise NotImplementedError("R mode [{:s}] is not supported.".format(R_mode))
        # multiply
        in_H, in_W, in_C = HSI.shape
        h_dim, w_dim, c_dim = 0, 1, 2
        if self.data_mode is None or self.data_mode == '':
            self.data_mode = 'HWC' if in_C < in_H else 'CHW'
        if self.data_mode == 'CHW':
            in_H, in_W, in_C = in_W, in_C, in_H
            h_dim, w_dim, c_dim = 1, 2, 0

        if c_dim == 2:
            out = torch.Tensor(in_H, in_W, R.size(0))
            for i in range(in_H):
                for j in range(in_W):
                    out[i, j, :] = R.mv(HSI[i, j, :])
        else:
            out = torch.Tensor(R.size(0), in_H, in_W)
            for i in range(in_H):
                for j in range(in_W):
                    out[:, i, j] = R.mv(HSI[:, i, j])
        return out


