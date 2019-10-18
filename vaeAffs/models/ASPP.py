import torch
import torch.nn as nn

class _ASPPModule3D(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, num_norm_groups):
        super(_ASPPModule3D, self).__init__()
        self.atrous_conv = nn.Conv3d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.GroupNorm(num_channels=planes, num_groups=num_norm_groups)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP3D(nn.Module):
    def __init__(self, inplanes, inner_planes, dilations, num_norm_groups=None):
        super(ASPP3D, self).__init__()

        aspp_modules = []
        aspp_modules.append(_ASPPModule3D(inplanes, inner_planes, 1, padding=0, dilation=1, num_norm_groups=num_norm_groups))
        for dil in dilations:
            aspp_modules.append(_ASPPModule3D(inplanes, inner_planes, 3, padding=dil, dilation=dil,
                                            num_norm_groups=num_norm_groups))
        self.aspp_modules = nn.ModuleList(aspp_modules)

        self.conv1 = nn.Conv3d(inner_planes*(len(dilations)+1), inner_planes, 1, bias=False)
        self.bn1 = nn.GroupNorm(num_channels=inner_planes, num_groups=num_norm_groups)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        intermediate = tuple(assp_module(x) for assp_module in self.aspp_modules)
        x = torch.cat(intermediate, dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # return self.dropout(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

