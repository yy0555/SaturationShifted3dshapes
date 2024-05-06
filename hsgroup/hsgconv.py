"""
© Felix O'Mahony
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

"""
This namespace is for the architecture of a (generalised) group convolutional network.

All classes in the network inherit nn.Module so that they may be included into a generalised PyTorch network.
In general modules accept and return data in the form (batch size, num groups * channels, *spatial dimensions) except special cases e.g. lifting.
The tensors are not separated into (batch size, num groups, channels, *spatial dimensions) because this allows the tensors to work with
functions which are not affected by group convolution (e.g. a 2D pool).

There are essentially X import classes in this namespace:

----------------
1. GroupConv
This group performs the group convolution.
The operation applied to the convolution filter between layers is defined by the filter_operation argument.
This should be a function with two inputs, x and k. x is the filter and k is the index of the filter which is being modified. 
$k \in \{0, \dots , n_groups - 1\}$.
$x \in \mathcal{R} ^{out channels, num groups, in channels, kernel size, kernel size}$
It should return a tensor with the same shape as x.
The default filter operation is to leave the filter unchanged (i.e. the identity operation).

By way of example, a function is defined spatial_rotation which performs the (standard) k * 90 degree spatial rotation of the filter.
This would be the appropriate filter for a spatial rotation equivariant group convolution.

----------------
2. GroupPool
This performs the final group pooling.

The method is a max pool, although this can be changed by editing the pool_operation function.

----------------

"""

class GroupConvHS(nn.Module):
    """
    Group Convolution Layer with hue and luminance group equivariance
    -----------------------
    This is a layer to be used within the global hue and luminance equivariance network. It is relatively simple, since no geometric transformation of the input tensor must take place. Rather, the input tensor has its group channels permuted so that each possible permutation of the color space (in hue, which we think of as rotation) and luminance (which we think of as scaling/permuting around radii of groups) occurs.

    This is described fully in the published paper.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        n_groups_hue=1,
        n_groups_saturation=1,
        bias = False,
        rescale_luminance = True,
        ) -> None:
        super().__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if type(self.kernel_size) != int:
            self.kernel_size = self.kernel_size[0]  # we only permit square kernels

        self.n_groups_hue = n_groups_hue
        self.n_groups_saturation = n_groups_saturation

        self.bias = bias

        self.rescale_luminance = rescale_luminance

        self.conv_weight = nn.Parameter(torch.Tensor(
            self.out_channels,
            self.n_groups_hue * self.n_groups_saturation * self.in_channels,
            self.kernel_size,
            self.kernel_size
        ))

        # Initialize the weights
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))

        self.construct_masks()

        self.register_buffer("group_conv_weight", self.construct_group_conv_weight())

    def construct_masks(self):
        mask = torch.zeros((self.n_groups_hue, self.n_groups_saturation, self.n_groups_hue, self.n_groups_saturation, 2), dtype=torch.int8)
        sfs = torch.ones((self.n_groups_hue, self.n_groups_saturation), dtype=torch.float32)
        group_elems_hue = torch.arange(self.n_groups_hue)
        group_elems_luminance = torch.arange(self.n_groups_saturation)
        H, L = torch.meshgrid(group_elems_hue, group_elems_luminance)
        group_elems = torch.stack((H, L), dim=-1)
        for i in range(self.n_groups_hue):
            for j in range(self.n_groups_saturation):
                roll_luminance = j - self.n_groups_saturation // 2
                mask[i, j, :, :] = group_elems.roll(i, dims=0).roll(roll_luminance, dims=1)
                if roll_luminance > 0:
                    mask[i, j, :, :roll_luminance, :] = -1
                elif roll_luminance < 0:
                    mask[i, j, :, roll_luminance:, :] = -1
                sfs[i, j] = (self.n_groups_saturation - abs(roll_luminance)) / self.n_groups_saturation

        self.mask = mask
        self.sfs = sfs

    def construct_group_conv_weight(self):
        conv_weight = torch.zeros((self.n_groups_hue, self.n_groups_saturation, self.out_channels, self.n_groups_hue, self.n_groups_saturation, self.in_channels, self.kernel_size, self.kernel_size), dtype=self.conv_weight.dtype)
        # put on same device as x
        conv_weight = conv_weight.to(self.conv_weight.device)
        cw = self.conv_weight.view(self.out_channels, self.n_groups_hue, self.n_groups_saturation, self.in_channels, self.kernel_size, self.kernel_size)
        for i in range(self.n_groups_hue):
            for j in range(self.n_groups_saturation):
                for k in range(self.n_groups_hue):
                    for l in range(self.n_groups_saturation):
                        if self.mask[i,j,k,l, 0] != -1:
                            conv_weight[i, j, :, k, l, :, :, :] = cw[:, self.mask[i,j,k,l, 0], self.mask[i,j,k,l, 1], :, :, :]
                            conv_weight[i, j, :, k, l, :, :, :] /= self.sfs[i, j]

        return conv_weight
                
        
    def forward_1(self, x):
        """
        incoming tensor is of shape (batch_size, n_groups_hue * n_groups_saturation * in_channels, height, width)
        outgoing tensor should be of shape (batch_size, n_groups_hue * n_groups_saturation * out_channels, height, width)
        """
        # reshape input tensor to shape appropriate for transforming according to hlgcnn
        x = x.view(-1, self.n_groups_hue, self.n_groups_saturation, self.in_channels, x.shape[-2], x.shape[-1])


        out_tensors = []
        for i in range(self.n_groups_hue):
            for j in range(self.n_groups_saturation):
                roll = j - self.n_groups_saturation // 2
                # remodel y to appropriately model x at this luminance
                y = x.roll(roll, dims=2)
                # set y values to zero where rolling pushes them through to other side
                if roll  > 0:
                    y[:, :, :roll, :, :, :] = torch.zeros_like(y[:, :, :roll, :, :, :])
                elif roll < 0:
                    y[:, :, roll:, :, :, :] = torch.zeros_like(y[:, :, roll:, :, :, :])
                    

                # Apply network
                # first we must reshape x to our target input
                y = y.view(-1, self.n_groups_hue * self.n_groups_saturation * self.in_channels, x.shape[-2], x.shape[-1])
                z = self.conv_layer(y)
                if self.rescale_luminance:
                    luminance_sf = (self.n_groups_saturation - abs(roll)) / self.n_groups_saturation
                    z /= luminance_sf
                out_tensors.append(z)
            x = x.roll(-1, dims=1)

        out_tensors = torch.stack(out_tensors, dim=1)
        out_tensors = out_tensors.view(-1, self.n_groups_hue * self.n_groups_saturation * self.out_channels, out_tensors.shape[-2], out_tensors.shape[-1])
        return out_tensors
    
    def forward_2(self, x):
        conv_weight = torch.zeros((self.n_groups_hue, self.n_groups_saturation, self.out_channels, self.n_groups_hue, self.n_groups_saturation, self.in_channels, self.kernel_size, self.kernel_size), dtype=self.conv_weight.dtype)
        # put on same device as x
        conv_weight = conv_weight.to(x.device)
        cw = self.conv_weight.view(self.out_channels, self.n_groups_hue, self.n_groups_saturation, self.in_channels, self.kernel_size, self.kernel_size)
        for i in range(self.n_groups_hue):
            for j in range(self.n_groups_saturation):
                roll_luminance = j - self.n_groups_saturation // 2
                conv_weight[i, j, :, :, :, :, :, :] = cw.roll(i, dims=1).roll(roll_luminance, dims=2)
                conv_weight[i, j, :, :, :, :, :, :] /= (self.n_groups_saturation - abs(roll_luminance)) / self.n_groups_saturation
                if roll_luminance > 0:
                    conv_weight[i, j, :, :, :roll_luminance, :, :, :] *= 0
                elif roll_luminance < 0:
                    conv_weight[i, j, :, :, roll_luminance:, :, :, :] *= 0

                # the mask has four dimensions and is boolean
                    # we want to assign the values of conv_weight[a, b, :, c, d, :, :, :] to cw[:, i, j, :,:,:]
                    # where mask[a, b, c, d] is True
                # we can do this by reshaping conv_weight to have shape (n_groups * n_groups_luminance, out_channels, n_groups * n_groups_luminance, in_channels, kernel_size, kernel_size)
                # and then reshaping cw to have shape (out_channels, n_groups * n_groups_luminance, in_channels, kernel_size, kernel_size)
                    

        conv_weight = conv_weight.view(self.n_groups_hue * self.n_groups_saturation * self.out_channels, self.n_groups_hue * self.n_groups_saturation * self.in_channels, self.kernel_size, self.kernel_size)
        out_tensors = F.conv2d(x, conv_weight, stride=self.stride, padding=self.padding)
        return out_tensors

    def forward_3(self,x):

        # out, in*n*groups, kernel, kernel
        # out*n_groups, in*n_groups, kernel, kernel
                    
        conv_weight = self.group_conv_weight

        conv_weight = conv_weight.view(self.n_groups_hue * self.n_groups_saturation * self.out_channels, self.n_groups_hue * self.n_groups_saturation * self.in_channels, self.kernel_size, self.kernel_size)
        out_tensors = F.conv2d(x, conv_weight, stride=self.stride, padding=self.padding)
        return out_tensors

    
    def forward(self, x):
        # out_tensors = self.forward_1(x)
        out_tensors = self.forward_2(x)
        # out_tensors = self.forward_3(x)

        return out_tensors

class GroupPool(nn.Module):
    def __init__(
        self, n_groups_total, pool_operation=lambda x: torch.max(x, dim=1)[0], verbose=False, name=None
    ) -> None:
        super().__init__()
        self.verbose = verbose
        self.n_groups = n_groups_total
        if verbose:
            print("verbose is sent to True in Pooling Layer")
        self.name = name if name is not None else "GroupPool"
        self.pool_operation = pool_operation

    def forward(self, x):
        x = x.view(
            -1, self.n_groups, x.shape[1] // self.n_groups, x.shape[2], x.shape[3]
        )

        # incoming tensor is of shape (batch_size, n_groups * channels, height, width)
        # outgoing tensor should be of shape (batch_size, channels, height, width)
        y = self.pool_operation(x)

        return y

class GroupBatchNorm2d(nn.Module):
    def __init__(self, num_features, n_groups_hue=4, n_groups_saturation=1, momentum = 0.1):
        super().__init__()
        self.batch_norm = nn.BatchNorm3d(num_features, momentum=momentum)
        self.num_features = num_features
        self.n_groups_hue = n_groups_hue
        self.n_groups_saturation = n_groups_saturation

    def forward(self, x):
        """
        incoming tensor is of shape (batch_size, n_groups * channels, height, width)"""
        if x.shape[1] != self.n_groups_hue * self.n_groups_saturation * self.num_features:
            raise ValueError(
                f"Expected {self.n_groups_hue * self.n_groups_saturation * self.num_features} channels in tensor, but got {x.shape[1]} channels"
            )
        x = x.view(
            -1, self.n_groups_hue, x.shape[-3] // (self.n_groups_hue * self.n_groups_saturation), x.shape[-2], x.shape[-1]
        )
        x = x.permute(0, 2, 1, 3, 4)
        y = self.batch_norm(x)
        y = y.permute(0, 2, 1, 3, 4)
        y = y.reshape(-1, (self.n_groups_hue * self.n_groups_saturation) * self.num_features, x.shape[-2], x.shape[-1])
        return y

if __name__=="__main__":
    # NB input tensor has shape (batch, groups_hue * groups_saturation * channels, w, h)
    test_input = torch.randn(1, 4 * 3 * 3, 32, 32)
    group_conv = GroupConvHS(3, 3, 3, n_groups_hue=4, n_groups_saturation=3)
    group_conv(test_input)