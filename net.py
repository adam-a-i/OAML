from collections import namedtuple
import torch
import torch.nn as nn
from torch.nn import Dropout
from torch.nn import MaxPool2d
from torch.nn import Sequential
from torch.nn import Conv2d, Linear
from torch.nn import BatchNorm1d, BatchNorm2d
from torch.nn import ReLU, Sigmoid
from torch.nn import Module
from torch.nn import PReLU
from qaconv import QAConv
import os
import torch.nn.functional as F


class OcclusionHead(Module):
    """Occlusion prediction head that predicts confidence maps for feature visibility.
    
    Takes feature maps from CNN backbone and outputs per-spatial-location occlusion confidence.
    Output values are in [0,1] where 1 = visible, 0 = occluded.
    """
    def __init__(self, in_channels, hidden_channels=256):
        """
        Args:
            in_channels: Number of input feature channels (e.g., 512 for IR-50)
            hidden_channels: Hidden layer channels for the prediction head
        """
        super(OcclusionHead, self).__init__()
        self.conv = Sequential(
            Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(hidden_channels),
            ReLU(inplace=True),
            Conv2d(hidden_channels, 1, kernel_size=1, bias=True),  # Single channel confidence
            Sigmoid()  # Values in [0,1]
        )
        
        # Initialize weights for stable training
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable occlusion prediction"""
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Feature maps [B, in_channels, H, W]
        
        Returns:
            occlusion_map: Confidence map [B, 1, H, W] where 1=visible, 0=occluded
        """
        return self.conv(x)


def build_model(model_name='ir_50'):
    if model_name == 'ir_101':
        return IR_101(input_size=(112,112))
    elif model_name == 'ir_50':
        return IR_50(input_size=(112,112))
    elif model_name == 'ir_se_50':
        return IR_SE_50(input_size=(112,112))
    elif model_name == 'ir_34':
        return IR_34(input_size=(112,112))
    elif model_name == 'ir_18':
        return IR_18(input_size=(112,112))
    else:
        raise ValueError('not a correct model name', model_name)

def initialize_weights(modules):
    """ Weight initilize, conv2d and linear is initialized with kaiming_normal
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()


class Flatten(Module):
    """ Flat tensor
    """
    def forward(self, input):
        return input.view(input.size(0), -1)


class LinearBlock(Module):
    """ Convolution block without no-linear activation layer
    """
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(LinearBlock, self).__init__()
        self.conv = Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class GNAP(Module):
    """ Global Norm-Aware Pooling block
    """
    def __init__(self, in_c):
        super(GNAP, self).__init__()
        self.bn1 = BatchNorm2d(in_c, affine=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn2 = BatchNorm1d(in_c, affine=False)

    def forward(self, x):
        x = self.bn1(x)
        x_norm = torch.norm(x, 2, 1, True)
        x_norm_mean = torch.mean(x_norm)
        weight = x_norm_mean / x_norm
        x = x * weight
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        feature = self.bn2(x)
        return feature


class GDC(Module):
    """ Global Depthwise Convolution block
    """
    def __init__(self, in_c, embedding_size):
        super(GDC, self).__init__()
        self.conv_6_dw = LinearBlock(in_c, in_c,
                                     groups=in_c,
                                     kernel=(7, 7),
                                     stride=(1, 1),
                                     padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(in_c, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size, affine=False)

    def forward(self, x):
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x


class SEModule(Module):
    """ SE block
    """
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction,
                          kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels,
                          kernel_size=1, padding=0, bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class OcclusionHead(Module):
    """Lightweight CNN head for spatial occlusion prediction.

    Predicts occlusion maps from backbone feature maps, where each spatial
    location indicates visibility confidence (1=visible, 0=occluded).

    Architecture:
        Conv2d(in_channels -> hidden_channels, 3x3) -> BN -> ReLU -> Conv2d(hidden_channels -> 1, 1x1) -> Sigmoid

    Args:
        in_channels: Number of input feature channels (default: 512 for IR networks)
        hidden_channels: Number of hidden channels (default: 128)

    Input:
        Feature maps of shape [B, in_channels, H, W] (e.g., [B, 512, 7, 7])

    Output:
        Occlusion maps of shape [B, 1, H, W] with values in [0, 1]
    """
    def __init__(self, in_channels=512, hidden_channels=128):
        super(OcclusionHead, self).__init__()

        # First conv: reduce channels and extract occlusion features
        self.conv1 = Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = BatchNorm2d(hidden_channels)
        self.relu = ReLU(inplace=True)

        # Second conv: predict single-channel occlusion map
        self.conv2 = Conv2d(hidden_channels, 1, kernel_size=1, bias=True)
        self.sigmoid = Sigmoid()

        # Initialize weights using Kaiming initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize conv layers with Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass to predict occlusion map.

        Args:
            x: Feature maps [B, C, H, W] from backbone

        Returns:
            Occlusion map [B, 1, H, W] with values in [0, 1]
            where 1 indicates visible regions and 0 indicates occluded regions
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x


class BasicBlockIR(Module):
    """ BasicBlock for IRNet
    """
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class BottleneckIR(Module):
    """ BasicBlock with bottleneck for IRNet
    """
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIR, self).__init__()
        reduction_channel = depth // 4
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, reduction_channel, (1, 1), (1, 1), 0, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, reduction_channel, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, depth, (1, 1), stride, 0, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class BasicBlockIRSE(BasicBlockIR):
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIRSE, self).__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))


class BottleneckIRSE(BottleneckIR):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIRSE, self).__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):

    return [Bottleneck(in_channel, depth, stride)] +\
           [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 18:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=2),
            get_block(in_channel=64, depth=128, num_units=2),
            get_block(in_channel=128, depth=256, num_units=2),
            get_block(in_channel=256, depth=512, num_units=2)
        ]
    elif num_layers == 34:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=6),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=256, num_units=3),
            get_block(in_channel=256, depth=512, num_units=8),
            get_block(in_channel=512, depth=1024, num_units=36),
            get_block(in_channel=1024, depth=2048, num_units=3)
        ]
    elif num_layers == 200:
        blocks = [
            get_block(in_channel=64, depth=256, num_units=3),
            get_block(in_channel=256, depth=512, num_units=24),
            get_block(in_channel=512, depth=1024, num_units=36),
            get_block(in_channel=1024, depth=2048, num_units=3)
        ]

    return blocks


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir'):
        """ Args:
            input_size: input_size of backbone
            num_layers: num_layers of backbone
            mode: support ir or irse
        """
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], \
            "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [18, 34, 50, 100, 152, 200], \
            "num_layers should be 18, 34, 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], \
            "mode should be ir or ir_se"
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64), PReLU(64))
        blocks = get_blocks(num_layers)
        if num_layers <= 100:
            if mode == 'ir':
                unit_module = BasicBlockIR
            elif mode == 'ir_se':
                unit_module = BasicBlockIRSE
            output_channel = 512
        else:
            if mode == 'ir':
                unit_module = BottleneckIR
            elif mode == 'ir_se':
                unit_module = BottleneckIRSE
            output_channel = 2048

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        # Store output_channel for reference
        self.output_channel = output_channel

        if input_size[0] == 112:
            self.output_layer = Sequential(BatchNorm2d(output_channel),
                                        Dropout(0.4), Flatten(),
                                        Linear(output_channel * 7 * 7, 512),
                                        BatchNorm1d(512, affine=False))
            self.qaconv = QAConv(num_features=output_channel, height=7, width=7)
            # Occlusion head for 112x112 input (7x7 feature maps)
            self.occlusion_head = OcclusionHead(in_channels=output_channel, hidden_channels=128)
        else:
            self.output_layer = Sequential(
                BatchNorm2d(output_channel), Dropout(0.4), Flatten(),
                Linear(output_channel * 14 * 14, 512),
                BatchNorm1d(512, affine=False))
            self.qaconv = QAConv(num_features=output_channel, height=14, width=14)
            # Occlusion head for 224x224 input (14x14 feature maps)
            self.occlusion_head = OcclusionHead(in_channels=output_channel, hidden_channels=128)

        initialize_weights(self.modules())
        
        # Register hooks for debugging
        if os.environ.get('DEBUG_LAYERS', '0') == '1':
            self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to debug NaN values in specific layers"""
        def hook_fn(module, input, output, layer_idx):
            if torch.isnan(output).any():
                print(f"NaN detected in layer {layer_idx} output")
                
        for i, module in enumerate(self.body):
            module.register_forward_hook(lambda mod, inp, out, idx=i: hook_fn(mod, inp, out, idx))

    def forward(self, x, return_occlusion=False):
        """
        Forward pass through backbone network.

        Args:
            x: Input images [B, 3, H, W]
            return_occlusion: If True, also return occlusion maps [B, 1, 7, 7]

        Returns:
            If return_occlusion=False (default):
                output: L2-normalized embeddings [B, 512]
                norm: Embedding norms [B, 1]
            If return_occlusion=True:
                output: L2-normalized embeddings [B, 512]
                norm: Embedding norms [B, 1]
                occlusion_map: Predicted occlusion maps [B, 1, H', W']
                feature_maps: Raw feature maps [B, C, H', W'] for QAConv
        """
        # Process through input layer
        x = self.input_layer(x)

        # Process through body with layer-by-layer NaN checks
        # Specifically fix problematic layers 22 and 23
        for idx, module in enumerate(self.body):
            x = module(x)

            # Check for NaNs after each body layer
            if torch.isnan(x).any():
                print(f"WARNING: Values after body layer {idx} contain NaNs. Replacing with zeros.")
                x = torch.nan_to_num(x, nan=0.0)

            # Apply special handling for layers 22-23 that are causing issues
            if idx in [21, 22, 23]:
                # Special stabilization for troublesome layers
                # 1. Check for extremely small values that might cause instability in future layers
                small_values_mask = (x.abs() < 1e-6) & (x != 0)
                if small_values_mask.any():
                    # Set very small non-zero values to a safe minimum to prevent potential division issues
                    x = torch.where(small_values_mask, torch.sign(x) * 1e-6, x)

                # 2. Check feature norm stability
                norms = torch.norm(x.view(x.size(0), x.size(1), -1), dim=2)
                if (norms < 1e-7).any():
                    # Apply channel-wise normalization to prevent collapse
                    x = F.layer_norm(x, [x.size(2), x.size(3)])

        # Check for NaNs or zeros in feature maps
        if torch.isnan(x).any() or (torch.sum(x.abs()) < 1e-4):
            print("WARNING: Feature maps contain NaNs or zeros before normalization. Fixing.")
            x = torch.nan_to_num(x, nan=0.0)
            # Apply stable normalization if needed
            if torch.sum(x.abs()) < 1e-4:
                x = x + 1e-6  # Add small constant to prevent all zeros

        feature_maps = x  # Store feature maps for QAConv and OcclusionHead

        # Check norms of feature maps to ensure they're reasonable
        norms = torch.norm(feature_maps.view(feature_maps.size(0), -1), p=2, dim=1)
        print(f"Feature map norms - min: {norms.min().item():.6f}, max: {norms.max().item():.6f}")

        # Only compute occlusion map when explicitly requested
        # This prevents unnecessary computation and BatchNorm stat updates on clean images
        occlusion_map = None
        if return_occlusion:
            occlusion_map = self.occlusion_head(feature_maps)  # [B, 1, H', W']

        # Get the output embedding
        embedding = self.output_layer(feature_maps)

        # Check for NaNs in embeddings
        if torch.isnan(embedding).any():
            print("WARNING: AdaFace embeddings contain NaNs. Replacing with zeros.")
            embedding = torch.nan_to_num(embedding, nan=0.0)

        # Safely normalize the final output
        norm = torch.norm(embedding, 2, 1, True).clamp(min=1e-6)  # Clamp to avoid division by zero
        output = torch.div(embedding, norm)

        # Return occlusion maps and feature maps if requested (for training)
        if return_occlusion:
            return output, norm, occlusion_map, feature_maps

        # QAConv matching score - only return during inference when gallery features are set
        if self.training == False and hasattr(self, '_gallery_features'):
            # Use feature maps directly without normalization to avoid introducing NaNs
            qaconv_score = self.qaconv(feature_maps, self._gallery_features)
            return output, norm, qaconv_score

        return output, norm

    def set_gallery_features(self, gallery_features):
        """Store gallery features for QAConv matching"""
        self._gallery_features = gallery_features


def IR_18(input_size):
    """ Constructs a ir-18 model.
    """
    model = Backbone(input_size, 18, 'ir')

    return model


def IR_34(input_size):
    """ Constructs a ir-34 model.
    """
    model = Backbone(input_size, 34, 'ir')

    return model


def IR_50(input_size):
    """ Constructs a ir-50 model.
    """
    model = Backbone(input_size, 50, 'ir')

    return model


def IR_101(input_size):
    """ Constructs a ir-101 model.
    """
    model = Backbone(input_size, 100, 'ir')

    return model


def IR_152(input_size):
    """ Constructs a ir-152 model.
    """
    model = Backbone(input_size, 152, 'ir')

    return model


def IR_200(input_size):
    """ Constructs a ir-200 model.
    """
    model = Backbone(input_size, 200, 'ir')

    return model


def IR_SE_50(input_size):
    """ Constructs a ir_se-50 model.
    """
    model = Backbone(input_size, 50, 'ir_se')

    return model


def IR_SE_101(input_size):
    """ Constructs a ir_se-101 model.
    """
    model = Backbone(input_size, 100, 'ir_se')

    return model


def IR_SE_152(input_size):
    """ Constructs a ir_se-152 model.
    """
    model = Backbone(input_size, 152, 'ir_se')

    return model


def IR_SE_200(input_size):
    """ Constructs a ir_se-200 model.
    """
    model = Backbone(input_size, 200, 'ir_se')

    return model

