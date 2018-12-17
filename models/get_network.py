import functools
import torch
import torch.nn as nn
from torch.nn import init
import models.architectures as arch


def choose(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_opt']
    which_model = opt_net['net_name']
    if which_model == 'UNet':
        net = arch.UNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'])
    elif which_model == 'DnCNN':
        net = arch.DnCNN(input_chnl=opt_net['in_nc'], num_chnl=opt_net['nf'])
    elif which_model == 'DPIR':
        net = arch.DPIR(input_chnl=opt_net['in_nc'], K=5, nf=opt_net['nf'])
    else:
        raise NotImplementedError('Choose network `{:s}` failed: NOT IMPLEMENTED.'.format(which_model))
    if gpu_ids:
        assert torch.cuda.is_available()
        net = nn.DataParallel(net)
    return net
