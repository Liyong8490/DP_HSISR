import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision
import models.get_network as get_network
from .base_model import BaseModel


class DenoisingModel(BaseModel):
    def __init__(self, opt):
        super(DenoisingModel, self).__init__(opt)
        train_opt = opt['train_opt']
        self.net = get_network.choose(opt).to(self.device)
        self.load()
        self.labels = None
        self.inputs = None
        self.preds = None

        if self.is_train:
            self.net.train()
            loss_type = train_opt['criterion']
            if loss_type == 'l1':
                self.cri = nn.L1Loss(size_average=False).to(self.device)
            elif loss_type == 'l2':
                self.cri = nn.MSELoss(size_average=False).to(self.device)
            else:
                raise NotImplementedError("Loss type {:s} is not support.".format(loss_type))

            # optimizer
            weight_decay = train_opt['weight_decay'] if train_opt['weight_decay'] else 0
            optim_params = []
            for k, v in self.net.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    print("WARNING: params [{:s}] will not optimize.".format(k))
            if str.upper(train_opt['optimizer']) == 'ADAM':
                self.optimizer = torch.optim.Adam(optim_params,
                                                  lr=train_opt['lr'],
                                                  weight_decay=weight_decay)
            elif str.upper(train_opt['optimizer']) == 'SGD':
                self.optimizer = torch.optim.SGD(optim_params,
                                                 lr=train_opt['lr'],
                                                 weight_decay=weight_decay)
            self.optimizers.append(self.optimizer)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer_ in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(
                        optimizer_,
                        train_opt['lr_steps'],
                        train_opt['lr_gamma']
                    ))
            else:
                raise NotImplementedError("lr_scheme: [{:s}] is not implemented."
                                          "Only MultiStepLR is supported now.".format(train_opt['lr_scheme']))
            self.log_dict = OrderedDict()

            print('---------- Model initialized ------------------')
            self.print_network()
            print('-----------------------------------------------')

    def feed_data(self, data, need_label=True):
        self.inputs = data['NI'].to(self.device)
        if need_label:
            self.labels = data['LB'].to(self.device)

    def optimize_parameters(self, ):
        if self.inputs is None or self.labels is None:
            raise ValueError("ERROR: inputs or labels is None. "
                             "Make sure to feed proper data to the model.")
        self.optimizer.zero_grad()
        self.preds = self.net(self.inputs)
        loss_ = self.cri(self.preds, self.labels) / (self.preds.data.shape[0] * 2)
        loss_.backward()
        self.optimizer.step()
        # set log
        self.log_dict['loss'] = loss_.item()

    def test(self):
        self.net.eval()
        with torch.no_grad():
            self.preds = self.net(self.inputs)
        self.net.train()

    def test_embed(self):
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.net.eval()
        for k, v in self.net.named_parameters():
            v.requires_grad = False

        def _transform(v_, op):
            # if self.precision != 'single': v = v.float()
            v2np = v_.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()
            else:
                raise NotImplementedError("transform type `{:s}` is nor implemented.".format(op))
            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()
            return ret

        lr_list = [self.inputs]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        sr_list = [self.net(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        self.preds = output_cat.mean(dim=0, keepdim=True)

        for k, v in self.net.named_parameters():
            v.requires_grad = True
        self.net.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_label=True):
        out_dict = OrderedDict()
        out_dict['inputs'] = self.inputs.detach()[0].float().cpu()
        out_dict['preds'] = self.preds.detach()[0].float().cpu()
        if need_label:
            out_dict['labels'] = self.labels.detach()[0].float().cpu()
        return out_dict

    def get_current_losses(self):
        return self.log_dict['loss']

    def print_network(self):
        s, n = self.get_network_description(self.net)
        print("Number of parameters in G: {:,d}".format(n))
        if self.is_train:
            message = "-------------- Network --------------\n" + s + "\n"
            network_path = os.path.join(self.save_dir, '../', 'network.txt')
            with open(network_path, 'w') as f:
                f.write(message)

    def load(self):
        load_path = self.opt['path_opt']['pretrained_model']
        if load_path is not None:
            print("loading model for network [{:s}] ...".format(load_path))
            self.load_network(load_path, self.net)

    def save(self, iter_label):
        self.save_network(self.save_dir, self.net, 'denoising_model', iter_label)


class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        train_opt = opt['train_opt']
        self.net = get_network.choose(opt).to(self.device)
        self.load()
        self.labels = None
        self.inputs = None
        self.preds = None

        if self.is_train:
            self.net.train()
            loss_type = train_opt['criterion']
            if loss_type == 'l1':
                self.cri = nn.L1Loss(size_average=False).to(self.device)
            elif loss_type == 'l2':
                self.cri = nn.MSELoss(size_average=False).to(self.device)
            else:
                raise NotImplementedError("Loss type {:s} is not support.".format(loss_type))

            # optimizer
            weight_decay = train_opt['weight_decay'] if train_opt['weight_decay'] else 0
            optim_params = []
            for k, v in self.net.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    print("WARNING: params [{:s}] will not optimize.".format(k))
            if str.upper(train_opt['optimizer']) == 'ADAM':
                self.optimizer = torch.optim.Adam(optim_params,
                                                  lr=train_opt['lr'],
                                                  weight_decay=weight_decay)
            elif str.upper(train_opt['optimizer']) == 'SGD':
                self.optimizer = torch.optim.SGD(optim_params,
                                                 lr=train_opt['lr'],
                                                 weight_decay=weight_decay)
            self.optimizers.append(self.optimizer)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer_ in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(
                        optimizer_,
                        train_opt['lr_steps'],
                        train_opt['lr_gamma']
                    ))
            else:
                raise NotImplementedError("lr_scheme: [{:s}] is not implemented."
                                          "Only MultiStepLR is supported now.".format(train_opt['lr_scheme']))
            self.log_dict = OrderedDict()

            print('---------- Model initialized ------------------')
            self.print_network()
            print('-----------------------------------------------')

    def feed_data(self, data, need_label=True):
        self.inputs = data['NI'].to(self.device)
        if need_label:
            self.labels = data['LB'].to(self.device)

    def optimize_parameters(self, ):
        if self.inputs is None or self.labels is None:
            raise ValueError("ERROR: inputs or labels is None. "
                             "Make sure to feed proper data to the model.")
        self.optimizer.zero_grad()
        self.preds = self.net(self.inputs)
        loss_ = self.cri(self.preds, self.labels) / (self.preds.data.shape[0] * 2)
        loss_.backward()
        self.optimizer.step()
        # set log
        self.log_dict['loss'] = loss_.item()

    def test(self):
        self.net.eval()
        with torch.no_grad():
            self.preds = self.net(self.inputs)
        self.net.train()

    def test_embed(self):
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.net.eval()
        for k, v in self.net.named_parameters():
            v.requires_grad = False

        def _transform(v_, op):
            # if self.precision != 'single': v = v.float()
            v2np = v_.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()
            else:
                raise NotImplementedError("transform type `{:s}` is nor implemented.".format(op))
            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()
            return ret

        lr_list = [self.inputs]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        sr_list = [self.net(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        self.preds = output_cat.mean(dim=0, keepdim=True)

        for k, v in self.net.named_parameters():
            v.requires_grad = True
        self.net.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_label=True):
        out_dict = OrderedDict()
        out_dict['inputs'] = self.inputs.detach()[0].float().cpu()
        out_dict['preds'] = self.preds.detach()[0].float().cpu()
        if need_label:
            out_dict['labels'] = self.labels.detach()[0].float().cpu()
        return out_dict

    def get_current_losses(self):
        return self.log_dict['loss']

    def print_network(self):
        s, n = self.get_network_description(self.net)
        print("Number of parameters in G: {:,d}".format(n))
        if self.is_train:
            message = "-------------- Network --------------\n" + s + "\n"
            network_path = os.path.join(self.save_dir, '../', 'network.txt')
            with open(network_path, 'w') as f:
                f.write(message)

    def load(self):
        load_path = self.opt['path_opt']['pretrained_model']
        if load_path is not None:
            print("loading model for network [{:s}] ...".format(load_path))
            self.load_network(load_path, self.net)

    def save(self, iter_label):
        self.save_network(self.save_dir, self.net, 'denoising_model', iter_label)


class FusionModel(BaseModel):
    def __init__(self, opt):
        super(FusionModel, self).__init__(opt)
        train_opt = opt['train_opt']
        self.net = get_network.choose(opt).to(self.device)
        self.load()
        self.labels = None
        self.inputs = None
        self.preds = None

        if self.is_train:
            self.net.train()
            loss_type = train_opt['criterion']
            if loss_type == 'l1':
                self.cri = nn.L1Loss(size_average=False).to(self.device)
            elif loss_type == 'l2':
                self.cri = nn.MSELoss(size_average=False).to(self.device)
            else:
                raise NotImplementedError("Loss type {:s} is not support.".format(loss_type))

            # optimizer
            weight_decay = train_opt['weight_decay'] if train_opt['weight_decay'] else 0
            optim_params = []
            for k, v in self.net.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    print("WARNING: params [{:s}] will not optimize.".format(k))
            if str.upper(train_opt['optimizer']) == 'ADAM':
                self.optimizer = torch.optim.Adam(optim_params,
                                                  lr=train_opt['lr'],
                                                  weight_decay=weight_decay)
            elif str.upper(train_opt['optimizer']) == 'SGD':
                self.optimizer = torch.optim.SGD(optim_params,
                                                 lr=train_opt['lr'],
                                                 weight_decay=weight_decay)
            self.optimizers.append(self.optimizer)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer_ in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(
                        optimizer_,
                        train_opt['lr_steps'],
                        train_opt['lr_gamma']
                    ))
            else:
                raise NotImplementedError("lr_scheme: [{:s}] is not implemented."
                                          "Only MultiStepLR is supported now.".format(train_opt['lr_scheme']))
            self.log_dict = OrderedDict()

            print('---------- Model initialized ------------------')
            self.print_network()
            print('-----------------------------------------------')

    def feed_data(self, data, need_label=True):
        self.inputs = data['NI'].to(self.device)
        if need_label:
            self.labels = data['LB'].to(self.device)

    def optimize_parameters(self, ):
        if self.inputs is None or self.labels is None:
            raise ValueError("ERROR: inputs or labels is None. "
                             "Make sure to feed proper data to the model.")
        self.optimizer.zero_grad()
        self.preds = self.net(self.inputs)
        loss_ = self.cri(self.preds, self.labels) / (self.preds.data.shape[0] * 2)
        loss_.backward()
        self.optimizer.step()
        # set log
        self.log_dict['loss'] = loss_.item()

    def test(self):
        self.net.eval()
        with torch.no_grad():
            self.preds = self.net(self.inputs)
        self.net.train()

    def test_embed(self):
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.net.eval()
        for k, v in self.net.named_parameters():
            v.requires_grad = False

        def _transform(v_, op):
            # if self.precision != 'single': v = v.float()
            v2np = v_.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()
            else:
                raise NotImplementedError("transform type `{:s}` is nor implemented.".format(op))
            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()
            return ret

        lr_list = [self.inputs]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        sr_list = [self.net(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        self.preds = output_cat.mean(dim=0, keepdim=True)

        for k, v in self.net.named_parameters():
            v.requires_grad = True
        self.net.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_label=True):
        out_dict = OrderedDict()
        out_dict['inputs'] = self.inputs.detach()[0].float().cpu()
        out_dict['preds'] = self.preds.detach()[0].float().cpu()
        if need_label:
            out_dict['labels'] = self.labels.detach()[0].float().cpu()
        return out_dict

    def get_current_losses(self):
        return self.log_dict['loss']

    def print_network(self):
        s, n = self.get_network_description(self.net)
        print("Number of parameters in G: {:,d}".format(n))
        if self.is_train:
            message = "-------------- Network --------------\n" + s + "\n"
            network_path = os.path.join(self.save_dir, '../', 'network.txt')
            with open(network_path, 'w') as f:
                f.write(message)

    def load(self):
        load_path = self.opt['path_opt']['pretrained_model']
        if load_path is not None:
            print("loading model for network [{:s}] ...".format(load_path))
            self.load_network(load_path, self.net)

    def save(self, iter_label):
        self.save_network(self.save_dir, self.net, 'denoising_model', iter_label)
