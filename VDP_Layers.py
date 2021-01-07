import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import grad

torch.set_default_tensor_type(torch.cuda.FloatTensor)
####################################################################################################
#
#   Author: Chris Angelini
#
#   Purpose: Expansion of the Bayesian VDP framework into Pytorch
#            The file is used for the creation of each layer:
#               Conv2D
#               RELU Activation
#               Maxpool2D
#               Fully Connected
#               Softmax
#
####################################################################################################

class VDP_Conv2D(nn.Module):
    """
    This class is for the instance creation of an Extended Variational Inference 2D Convolutional
    Layer. This class contains the function :func:`__init__` for the initialization of the instance,
    the function :func:`forward` for the forward propagation through the layer when called, and the
    last function :func:`kl_loss_term` which is called in the loss function as part of the
    regularization of the network.
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size=5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, padding_mode='zeros',
                 mean_mu=0, mean_sigma=0.1, sigma_min=-12, sigma_max=-2.2,
                 input_flag=False):
        """
        :param in_channels:  Number of input channels        (Required)
        :param out_channels: Number of output channels       (Required)
        :param kernel_size:  Size of the conv, kernel        (Default   5)
        :param stride:       Stride Length                   (Default   1)
        :param padding:      Padding  T/F                    (Default   0)
        :param dilation:     Dilation T/F                    (Default   1)
        :param groups:       Groups                          (Default   1)
        :param bias:         Bias T/F                        (Default   False)
        :param padding_mode: Padding_mode                    (Default  'zeros')
        :param mean_mu:      Mean  Weight Init. Distr. mu    (Default   0)
        :param mean_sigma:   Mean  Weight Init. Distr. sigma (Default   0.1)
        :param sigma_min:    Sigma Weight Init. Distr. mu    (Default  -12)
        :param sigma_max:    Sigma Weight Init. Distr. sigma (Default  -2.2)
        :param input_flag                                    (Default   False)
        """
        super(VDP_Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_flag = input_flag

        self.mean_conv = nn.Conv2d(in_channels, out_channels,
                                   kernel_size, stride, padding, dilation,
                                   groups, bias, padding_mode)
        nn.init.normal_(self.mean_conv.weight, mean=mean_mu, std=mean_sigma)

        self.sigma_conv_weight = nn.Parameter(torch.zeros([1, out_channels]), requires_grad=True)
        nn.init.uniform_(self.sigma_conv_weight, a=sigma_min, b=sigma_max)

        self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)

    def forward(self, mu, sigma=0):
        """
        Forward pass over the VDP 2D Convolutional Layer

        :param mu:      Data Mean     (Required)
        :type  mu:      Float
        :param sigma:   Data Sigma    (Required only with input_flag=False)
        :type  sigma:   Float
        """
        if self.input_flag:
            # Input Version
            mu_z = self.mean_conv(mu)

            x_patches = self.unfold(mu).permute(0, 2, 1)
            x_matrix = torch.bmm(x_patches, x_patches.permute(0, 2, 1))
            x_matrix_tile = x_matrix.unsqueeze(1).repeat(1, self.out_channels, 1, 1)
            sigma_z = torch.mul(torch.log(1. + torch.exp(self.sigma_conv_weight)), x_matrix_tile.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            return mu_z, sigma_z
        else:
            # Non-Input Version
            mu_z = self.mean_conv(mu)

            diag_sigma = torch.diagonal(sigma, dim1=2, dim2=3)
            sigma_patches = self.unfold(diag_sigma.reshape([mu.shape[0], mu.shape[1], mu.shape[2], mu.shape[2]]))

            mu_cov = torch.reshape(self.mean_conv.weight, [-1, self.kernel_size * self.kernel_size * mu.shape[1]])
            mu_cov_square = mu_cov * mu_cov
            mu_wT_sigma_mu_w1 = torch.matmul(mu_cov_square, sigma_patches)
            sigma_1 = torch.diag_embed(mu_wT_sigma_mu_w1, dim1=2)

            trace = torch.sum(sigma_patches, 1).unsqueeze(2).repeat(1, 1, self.out_channels)
            trace = torch.mul(torch.log(1 + torch.exp(self.sigma_conv_weight)), trace).permute(0, 2, 1)
            trace_1 = torch.diag_embed(trace, dim1=2)

            x_patches = self.unfold(mu).permute(0, 2, 1)
            x_matrix = torch.bmm(x_patches, x_patches.permute(0, 2, 1))
            x_matrix_tile = x_matrix.unsqueeze(1).repeat(1, self.out_channels, 1, 1)
            sigma_3 = torch.mul(torch.log(1. + torch.exp(self.sigma_conv_weight)), x_matrix_tile.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            sigma_z = sigma_1 + trace_1 + sigma_3

            return mu_z, sigma_z

    def kl_loss_term(self):
        """
        KL Loss term for the loss function
        """
        weights_mean = self.mean_conv.weight.permute([2, 3, 1, 0])

        c_s = torch.log(1. + torch.exp(self.sigma_conv_weight))

        kl_loss = -0.5 * torch.mean(self.kernel_size * self.kernel_size + (self.kernel_size * self.kernel_size) * torch.log(c_s) -
                                    torch.abs(weights_mean).sum() - (self.kernel_size * self.kernel_size) * c_s)
        return kl_loss


class VDP_Relu(nn.Module):
    """
    This class is for the instance creation of an Extended Variational Inference RELU. This
    class contains the function :func:`__init__` for the initialization of the instance and
    the function :func:`forward` for the forward propagation through the layer when called
    """

    def __init__(self, inplace=False):
        """
        :param inplace:         (Default  False)
        """
        super(VDP_Relu, self).__init__()
        self.relu = nn.SELU(inplace)

    def forward(self, mu, sigma):
        """
        Forward pass over the VDP Relu Layer

        :param mu:      Data Mean     (Required)
        :type  mu:      Float
        :param sigma:   Data Sigma    (Required)
        :type  sigma:   Float
        """
        mu_g = self.relu(mu)

        activation_gradient = grad(mu_g.sum(), mu, retain_graph=True)[0]

        if len(mu_g.shape) == 2:

            grad_square = torch.bmm(activation_gradient.unsqueeze(2), activation_gradient.unsqueeze(1))

            sigma_g = torch.mul(sigma, grad_square).unsqueeze(1)
        else:
            gradient_matrix = activation_gradient.permute([0, 2, 3, 1]).view(activation_gradient.shape[0], -1, mu_g.shape[1]).unsqueeze(3)

            grad1 = gradient_matrix.permute([0, 2, 1, 3])
            grad2 = grad1.permute([0, 1, 3, 2])

            grad_square = torch.matmul(grad1, grad2)

            sigma_g = torch.mul(sigma, grad_square)  # shape =[image_size*image_size,image_size*image_size, num_filters[0]]
        return mu_g, sigma_g


class VDP_Maxpool(nn.Module):
    """
    This class is for the instance creation of an Extended Variational Inference 2D Maxpool. This
    class contains the function :func:`__init__` for the initialization of the instance and
    the function :func:`forward` for the forward propagation through the layer when called
    """

    def __init__(self, kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True, ceil_mode=False):
        """
        :param kernel_size:     Kernel Size     (Default  2)
        :param stride:          Stride Length   (Default  2)
        :param padding:         Padding T/F     (Default  0)
        :param dilation:        Dilation        (Default  1)
        :param return_indices:  Return Indices  (Default  True)
        :param ceil_mode:       Ceiling Mode    (Default  False)
        """
        super(VDP_Maxpool, self).__init__()
        self.maxPooling = nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def forward(self, mu, sigma):
        """
        Forward pass over the VDP 2D Maxpool Layer

        :param mu:      Data Mean     (Required)
        :type  mu:      Float
        :param sigma:   Data Sigma    (Required)
        :type  sigma:   Float
        """
        mu_p, argmax1 = self.maxPooling(mu)

        argmax = argmax1.reshape(argmax1.shape[0], argmax1.shape[1], argmax1.shape[2] ** 2)

        argmax_indices = argmax.repeat(1, 1, argmax.shape[-1]).reshape(argmax.shape[0], argmax.shape[1], argmax.shape[-1], argmax.shape[-1])

        index_fix = argmax.unsqueeze(3) * sigma.shape[-1]
        sigma_indexes = argmax_indices + index_fix

        sigma_p = torch.gather(sigma.reshape(argmax.shape[0], argmax.shape[1], -1),
                               dim=2,
                               index=sigma_indexes.reshape(argmax.shape[0], argmax.shape[1], -1)).reshape(argmax.shape[0],  argmax.shape[1],
                                                                                                          argmax.shape[2], argmax.shape[2])
        return mu_p, sigma_p


class VDP_FullyConnected(nn.Module):
    """
    This class is for the instance creation of an Extended Variational Inference Fully Connected
    Layer. This class contains the function :func:`__init__` for the initialization of the instance,
    the function :func:`forward` for the forward propagation through the layer when called, and
    the last function :func:`kl_loss_term` which is called in the loss function as part of the
    regularization of the network.
    """

    def __init__(self, in_features, out_features, bias=True,
                 mean_mu=0, mean_sigma=0.1, sigma_min=-12, sigma_max=-2.2,
                 mean_bias=0.001, sigma_bias=0.001,
                 input_flag=False):
        """
        :param in_features:     (Required)
        :param out_features:    (Required)
        :param bias:            (Default  True)
        :param mean_mu:         (Default  0)
        :param mean_sigma:      (Default  0.1)
        :param sigma_min:       (Default -12)
        :param sigma_max:       (Default -2.2)
        :param mean_bias:       (Default  0.001)
        :param sigma_bias:      (Default  0.001)
        :param input_flag:      (Default  False)
        """
        super(VDP_FullyConnected, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.input_flag = input_flag

        self.mean_fc = nn.Linear(in_features, out_features, bias)

        nn.init.normal_(self.mean_fc.weight, mean=mean_mu, std=mean_sigma)
        self.mean_fc.bias.data.fill_(mean_bias)

        self.sigma_fc_weight = nn.Parameter(torch.zeros([1, out_features]), requires_grad=True)
        nn.init.uniform_(self.sigma_fc_weight, a=sigma_min, b=sigma_max)
        self.sigma_fc_bias = nn.Parameter(torch.tensor(sigma_bias), requires_grad=True)

    def forward(self, mu, sigma=0):
        """
        Forward pass over the VDP 2D Fully Connected Layer

        :param mu:      Data Mean     (Required)
        :type  mu:      Float
        :param sigma:   Data Sigma    (Required if input_flag=False)
        :type  sigma:   Float
        """
        if self.input_flag:
            # Input Version
            mu_f = self.mean_fc(mu)

            mu_pT_mu_p = (mu * mu).reshape(-1, mu.shape[1:4].numel()).sum(1)
            mu_pT_sigma_h_mu_p = torch.log(1. + torch.exp(self.sigma_fc_weight)).repeat(mu.shape[0], 1) * mu_pT_mu_p.unsqueeze(1)
            mu_pT_sigma_h_mu_p = torch.diag_embed(mu_pT_sigma_h_mu_p, dim1=1)

            sigma_f = mu_pT_sigma_h_mu_p

        else:
            # Non-Input Version
            if len(sigma.shape) == 3:
                sigma = sigma.unsqueeze(1)

            diag_elements = torch.diagonal(sigma, dim1=2, dim2=3)
            diag_sigma_b = diag_elements.reshape([mu.shape[0], -1])

            mu_f = self.mean_fc(mu)

            fc_weight_mean1 = self.mean_fc.weight.T.reshape([sigma.shape[1], sigma.shape[2], self.out_features])
            fc_weight_mean1T = fc_weight_mean1.permute([0, 2, 1])
            muhT_sigmab = torch.matmul(fc_weight_mean1T, sigma)  # OLD THING AND ISSUE
            muhT_sigmab_mu = torch.matmul(muhT_sigmab, fc_weight_mean1).sum(1)

            tr_sigma_b = diag_sigma_b.sum(1)  # NEW INPUT
            tr_sigma_h_sigma_b = torch.log(1. + torch.exp(self.sigma_fc_weight)).repeat(mu.shape[0], 1) * tr_sigma_b.unsqueeze(1)
            tr_sigma_h_sigma_b = torch.diag_embed(tr_sigma_h_sigma_b, dim1=1)

            mu_pT_mu_p = (mu * mu).reshape(-1, mu.shape[1:4].numel()).sum(1)
            mu_pT_sigma_h_mu_p = torch.log(1. + torch.exp(self.sigma_fc_weight)).repeat(mu.shape[0], 1) * mu_pT_mu_p.unsqueeze(1)
            mu_pT_sigma_h_mu_p = torch.diag_embed(mu_pT_sigma_h_mu_p, dim1=1)

            sigma_f = muhT_sigmab_mu + tr_sigma_h_sigma_b + mu_pT_sigma_h_mu_p

        return mu_f, sigma_f

    def kl_loss_term(self):
        """
        KL Loss term for the loss function
        """
        f_s = torch.log(1 + torch.exp(self.sigma_fc_weight))

        kl_loss = -0.5 * torch.mean((self.in_features * torch.log(f_s)) -
                                    torch.abs(self.mean_fc.weight).sum() -
                                    (self.in_features * f_s))
        return kl_loss


class VDP_Softmax(nn.Module):
    """
    This class is for the instance creation of an Extended Variational Inference Softmax Activation.
    This class contains the function :func:`__init__` for the initialization of the instance
    and the function :func:`forward` for the forward propagation through the layer when called
    """

    def __init__(self, dim=1):
        """
        :param dim:     (Default  1)
        """
        super(VDP_Softmax, self).__init__()
        self.softmax_mu_y = nn.Softmax(dim)

    def forward(self, mu, sigma):
        """
        Forward pass over the VDP 2D Softmax Layer

        :param mu:      Data Mean     (Required)
        :type  mu:      Float
        :param sigma:   Data Sigma    (Required)
        :type  sigma:   Float
        """
        mu_y = self.softmax_mu_y(mu)

        grad_f1 = torch.bmm(mu_y.unsqueeze(2), mu_y.unsqueeze(1))
        diag_f = torch.diag_embed(mu_y, dim1=1)
        grad_soft = diag_f - grad_f1
        sigma_y = torch.matmul(grad_soft, torch.matmul(sigma, grad_soft.permute(0, 2, 1)))
        return mu_y, sigma_y