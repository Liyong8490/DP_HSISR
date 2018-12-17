from models.blocks import *


class DnCNN(nn.Module):
    def __init__(self, input_chnl, num_chnl=64, groups=1):
        super(DnCNN, self).__init__()

        kernel_size = 3

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=input_chnl, out_channels=num_chnl,
                                             kernel_size=kernel_size, stride=1, padding=1,
                                             groups=1, bias=True),
                                   nn.ReLU(inplace=True))
        self.dn_block = self._make_layers(Conv_BN_ReLU, kernel_size, num_chnl, num_of_layers=15, bias=False)
        self.output = nn.Conv2d(in_channels=num_chnl, out_channels=input_chnl,
                                kernel_size=kernel_size, stride=1, padding=1,
                                groups=groups, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if 0 <= m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif -clip_b < m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)

    def _make_layers(self, block, kernel_size, num_chnl, num_of_layers, padding=1, groups=1, bias=False):
        layers = []
        for _ in range(num_of_layers):
            layers.append(block(in_channels=num_chnl, out_channels=num_chnl, kernel_size=kernel_size, padding=padding,
                                groups=groups, bias=bias))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        # x = self.nl(x)
        x = self.dn_block(x)
        return self.output(x) + residual


class DPIR(nn.Module):
    """
    Denoising Prior Driven Deep Neural Network for Image Restoration
    Weisheng Dong, Peiyao Wang, Wotao Yin, et. al. ArXiv: 1801.06756
    Image Restoration formulation:
                x,v = argmax_(x,v) 0.5*|y-Ax|^2 + lambda*J(v)
                        s.t. x=v
            apply half-quadratic splitting method:
                x,v = argmax_(x,v) 0.5*|y-Ax|^2 + eta*|x-v|^2 + lambda*J(v)
            following two sub-problems:
                1: x = argmax_x |y - Ax|^2 + eta*|x - v|^2
                2: v = argmax_v eta*|x - v|^2 + lambda*J(v)
            where sub-problem 2 is a typical denoising problem which can be replaced by a CNN denoiser.
    Using conjugate gradient algorithm to solve sub-problem 1:
            x   = x - delta*[A'(Ax - y) - eta*(x - v)]
                = A_t*x + delta*A'y + delta*v
            where A_t = [(1 - delta*eta)I - delta*A'A]
    In the problem of denoising:
            A is selected as a identify operator. Thus:
                A_t = (1 - delta*(eta + 1))I ,
                A' = A = I .
    """
    def __init__(self, input_chnl, K=5, nf=64, groups=1):
        super(DPIR, self).__init__()
        self.K = K
        self.denoiser = DnCNN(input_chnl, nf, groups)
        self.delta = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.eta = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.delta.data.uniform_(0.01, 0.1)
        self.eta.data.uniform_(0.01, 0.1)

    def forward(self, y):
        delta_AT_y = self.delta * y
        A_t = 1 - self.delta * (self.eta + 1)
        x = A_t * y + delta_AT_y
        for _ in range(self.K):
            v = self.denoiser(x)
            delta_v = self.delta * v
            x = A_t * x + delta_AT_y + delta_v
        return x


class DPHSISR(nn.Module):
    def __init__(self, in_c, out_c, nf):
        super(DPHSISR, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.nf = nf

    def forward(self, x):
        return x
