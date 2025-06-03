import torch
import torch.nn as nn
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import BayesianRidge

from .resnet import resnet18, resnet18_nopool, BasicBlock
from .conformer import ConformerBlock
import math
import numpy as np

layer_resnet = ['conv1', 'bn1', 'relu', 'layer1', 'layer1.0', 'layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu', 'layer1.0.conv2', 'layer1.0.bn2', 'layer1.1', 'layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.relu', 'layer1.1.conv2', 'layer1.1.bn2', 'maxpool1', 'layer2', 'layer2.0', 'layer2.0.conv1', 'layer2.0.bn1', 'layer2.0.relu', 'layer2.0.conv2', 'layer2.0.bn2', 'layer2.0.downsample', 'layer2.0.downsample.0', 'layer2.0.downsample.1', 'layer2.1', 'layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.relu', 'layer2.1.conv2', 'layer2.1.bn2', 'maxpool2', 'layer3', 'layer3.0', 'layer3.0.conv1', 'layer3.0.bn1', 'layer3.0.relu', 'layer3.0.conv2', 'layer3.0.bn2', 'layer3.0.downsample', 'layer3.0.downsample.0', 'layer3.0.downsample.1', 'layer3.1', 'layer3.1.conv1', 'layer3.1.bn1', 'layer3.1.relu', 'layer3.1.conv2', 'layer3.1.bn2', 'maxpool3', 'layer4', 'layer4.0', 'layer4.0.conv1', 'layer4.0.bn1', 'layer4.0.relu', 'layer4.0.conv2', 'layer4.0.bn2', 'layer4.0.downsample', 'layer4.0.downsample.0', 'layer4.0.downsample.1', 'layer4.1', 'layer4.1.conv1', 'layer4.1.bn1', 'layer4.1.relu', 'layer4.1.conv2', 'layer4.1.bn2', 'conv5']

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# Channel Shuffle Block
class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, dimn1,dimn2 = x.size()
        assert num_channels % self.groups == 0, "Channels not divisible by group size"
        
        x = x.view(batch_size, self.groups, num_channels // self.groups, dimn1,dimn2)
        
        x = x.permute(0, 2, 1, 3, 4)
        
        x = x.reshape(batch_size, num_channels, dimn1,dimn2)
        return x

# Residual GConv Block
class ResidualGConvBlock(nn.Module):
    def __init__(self, in_channels, groups=2):
        super(ResidualGConvBlock, self).__init__()
        self.gconv = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shuffle = ChannelShuffle(groups)

    def forward(self, x):
        identity = x  # Residual connection
        out = self.gconv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.shuffle(out)
        out += identity  # Add residual
        return out

# SED_DOA_3 with GConv and Channel Shuffle before the second conformer block
class SED_DOA_5(nn.Module):
    def __init__(self, in_channel, in_dim):
        super().__init__()
        self.resnet = resnet18_nopool(in_channel=in_channel)
        embedding_dim = in_dim // 32 * 256
        encoder_dim = 256

        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, encoder_dim),
            nn.Dropout(p=0.05),
        )

        num_layers = 6
        self.conformer_layers = nn.ModuleList([
            ConformerBlock(
                dim=encoder_dim,
                dim_head=32,
                heads=8,
                ff_mult=2,
                conv_expansion_factor=2,
                conv_kernel_size=7,
                attn_dropout=0.1,
                ff_dropout=0.1,
                conv_dropout=0.1
            ) for _ in range(num_layers)
        ])

        self.t_pooling = nn.MaxPool1d(kernel_size=5)

        # GConv + Channel Shuffle before the second set of conformers
        self.gconv_shuffle = ResidualGConvBlock(in_channels=50, groups=2)

        self.input_projection_2 = nn.Sequential(
            nn.Linear(encoder_dim + 1536, encoder_dim),
            nn.Dropout(p=0.05),
        )

        num_layers_2 = 2
        self.conformer_layers_2 = nn.ModuleList([
            ConformerBlock(
                dim=encoder_dim,
                dim_head=32,
                heads=8,
                ff_mult=2,
                conv_expansion_factor=2,
                conv_kernel_size=7,
                attn_dropout=0.1,
                ff_dropout=0.1,
                conv_dropout=0.1
            ) for _ in range(num_layers_2)
        ])

        self.sed_out_layer = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.LeakyReLU(),
            nn.Linear(encoder_dim, 13),
            nn.Sigmoid()
        )

        self.out_layer = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.LeakyReLU(),
            nn.Linear(encoder_dim, 26),
            nn.Tanh()
        )

    def forward(self, x, onepeace):
        # Initial feature extraction
        # print("Input features shape: ", x.shape)
        # print("Onepeace Embeddings shape:", onepeace.shape)
        conv_outputs = self.resnet(x)
        # print("After Resnet: ", conv_outputs.shape)
        N, C, T, W = conv_outputs.shape
        conv_outputs = conv_outputs.permute(0, 2, 1, 3).reshape(N, T, C * W)
        # print("After Resnet's output's reshape: ", conv_outputs.shape)
        # First Conformer block processing
        conformer_outputs = self.input_projection(conv_outputs)
        # print("After projection (before sending into first conformer block): ", conformer_outputs.shape)
        for layer in self.conformer_layers:
            conformer_outputs = layer(conformer_outputs)
        # print("After first conformer block: ", conformer_outputs.shape)
        # Temporal pooling
        outputs = conformer_outputs.permute(0, 2, 1)
        # print("After first conformer's output's reshape: ", outputs.shape)
        outputs = self.t_pooling(outputs)
        # print("After time pooling: ", outputs.shape)
        outputs = outputs.permute(0, 2, 1)
        # print("After pooling's output's reshape: ", outputs.shape)
        # Fusion with onepeace embeddings
        onepeace = onepeace.squeeze(1)
        # print("After onepeace's  reshape: ", onepeace.shape)
        outputs = torch.cat((outputs, onepeace), dim=-1)
        # print("After concatenation: ", outputs.shape)

        # GConv + Channel Shuffle (with residual)
        outputs = outputs.unsqueeze(-2) # Adding dimensions for Conv2d
        # print("After concatenation's output's reshape (for adapting to gconv format): ", outputs.shape)
        outputs = self.gconv_shuffle(outputs)
        # print("After gconv+shuffle: ", outputs.shape)
        outputs = outputs.squeeze(-2) # Removing added dimensions
        # print("After gconv+shuffle's output's reshape: ", outputs.shape)
        outputs = self.input_projection_2(outputs)
        # print("After projection (before sending into second conformer block):", outputs.shape)
        # Second Conformer block processing
        for layer_2 in self.conformer_layers_2:
            outputs = layer_2(outputs)
        # print("After second conformer block:", outputs.shape)
        # Prediction layers
        sed = self.sed_out_layer(outputs)
        # print("After sed head: ", sed.shape)
        doa = self.out_layer(outputs)
        # print("After doa head: ", doa.shape)
        pred = torch.cat((sed, doa), dim=-1)  # [N, T, 39]
        # print("Final prediction: ", pred.shape)
        return pred

