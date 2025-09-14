import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, List


class DoubleConv(nn.Module):
    """Double convolution block with batch norm and ReLU"""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DenseBlock(nn.Module):
    """Dense block for UNet++ light mode"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x, skip_connections: List[torch.Tensor]):
        # Concatenate input with all skip connections
        if skip_connections:
            x = torch.cat([x] + skip_connections, dim=1)
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    """UNet++ Light with selectable encoder and dense skip connections"""
    
    def __init__(self, 
                 encoder_name: str = "resnet34",
                 in_channels: int = 3,
                 num_classes: int = 1,
                 features: List[int] = [64, 128, 256, 512]):
        super().__init__()
        
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = features
        
        # Encoder selection
        if encoder_name == "resnet34":
            self.encoder = models.resnet34(pretrained=True)
            encoder_channels = [64, 64, 128, 256, 512]
        elif encoder_name == "resnet50":
            self.encoder = models.resnet50(pretrained=True)
            encoder_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}. Use 'resnet34' or 'resnet50'")
        
        # Remove classifier and pooling
        self.encoder = nn.ModuleList([
            nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu),
            self.encoder.layer1,
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4
        ])
        
        # Decoder with dense connections
        self.decoder = nn.ModuleDict()
        
        # Level 0 (deepest)
        self.decoder['0_0'] = DenseBlock(encoder_channels[4], features[3])
        
        # Level 1
        self.decoder['1_0'] = DenseBlock(encoder_channels[3], features[2])
        self.decoder['1_1'] = DenseBlock(features[3] + features[2], features[2])
        
        # Level 2
        self.decoder['2_0'] = DenseBlock(encoder_channels[2], features[1])
        self.decoder['2_1'] = DenseBlock(features[2] + features[1], features[1])
        self.decoder['2_2'] = DenseBlock(features[2] + features[1] + features[1], features[1])
        
        # Level 3
        self.decoder['3_0'] = DenseBlock(encoder_channels[1], features[0])
        self.decoder['3_1'] = DenseBlock(features[1] + features[0], features[0])
        self.decoder['3_2'] = DenseBlock(features[1] + features[0] + features[0], features[0])
        self.decoder['3_3'] = DenseBlock(features[1] + features[0] + features[0] + features[0], features[0])
        
        # Final output
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        # Encoder path
        encoder_outputs = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            encoder_outputs.append(x)
        
        # Decoder path with dense connections
        # Level 0
        x0_0 = self.decoder['0_0'](encoder_outputs[4], [])
        
        # Level 1
        x1_0 = self.decoder['1_0'](encoder_outputs[3], [])
        x1_1 = self.decoder['1_1'](self.upsample(x0_0), [x1_0])
        
        # Level 2
        x2_0 = self.decoder['2_0'](encoder_outputs[2], [])
        x2_1 = self.decoder['2_1'](self.upsample(x1_0), [x2_0])
        x2_2 = self.decoder['2_2'](self.upsample(x1_1), [x2_0, x2_1])
        
        # Level 3
        x3_0 = self.decoder['3_0'](encoder_outputs[1], [])
        x3_1 = self.decoder['3_1'](self.upsample(x2_0), [x3_0])
        x3_2 = self.decoder['3_2'](self.upsample(x2_1), [x3_0, x3_1])
        x3_3 = self.decoder['3_3'](self.upsample(x2_2), [x3_0, x3_1, x3_2])
        
        # Final output (use the deepest dense connection)
        output = self.final_conv(x3_3)
        
        return output


class UNet(nn.Module):
    """Standard UNet with selectable encoder"""
    
    def __init__(self, 
                 encoder_name: str = "resnet34",
                 in_channels: int = 3,
                 num_classes: int = 1,
                 features: List[int] = [64, 128, 256, 512]):
        super().__init__()
        
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = features
        
        # Encoder selection
        if encoder_name == "resnet34":
            self.encoder = models.resnet34(pretrained=True)
            encoder_channels = [64, 64, 128, 256, 512]
        elif encoder_name == "resnet50":
            self.encoder = models.resnet50(pretrained=True)
            encoder_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}. Use 'resnet34' or 'resnet50'")
        
        # Remove classifier and pooling
        self.encoder = nn.ModuleList([
            nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu),
            self.encoder.layer1,
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4
        ])
        
        # Decoder with proper channel handling
        self.decoder = nn.ModuleList()
        # Build 4 decoding stages: (512+256)->256, (256+128)->128, (128+64)->64, (64+64?)->64 is not used; stop at layer1
        for i in range(len(encoder_channels) - 1, 0, -1):
            in_channels_decoder = encoder_channels[i] + encoder_channels[i - 1]
            out_channels_decoder = encoder_channels[i - 1]
            self.decoder.append(
                nn.Sequential(
                    DoubleConv(in_channels_decoder, out_channels_decoder),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
            )
        
        # Final output
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder path
        encoder_outputs = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            encoder_outputs.append(x)
        
        # Decoder path
        x = encoder_outputs[-1]
        # Iterate over decoder stages; each stage uses the corresponding skip from encoder_outputs
        for stage_idx, decoder_layer in enumerate(self.decoder):
            # target encoder index for skip: len-2, len-3, ..., 0
            enc_skip_idx = len(encoder_outputs) - 2 - stage_idx
            x = F.interpolate(x, size=encoder_outputs[enc_skip_idx].shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, encoder_outputs[enc_skip_idx]], dim=1)
            x = decoder_layer(x)
        
        # Final output
        output = self.final_conv(x)
        
        return output


def create_unet_model(encoder_name: str = "resnet34", 
                     unet_plus_plus: bool = True,
                     in_channels: int = 3,
                     num_classes: int = 1) -> nn.Module:
    """
    Create UNet or UNet++ model with specified encoder
    
    Args:
        encoder_name: 'resnet34' or 'resnet50'
        unet_plus_plus: If True, use UNet++ light mode, else standard UNet
        in_channels: Number of input channels
        num_classes: Number of output classes (1 for binary)
    
    Returns:
        UNet or UNet++ model
    """
    if unet_plus_plus:
        return UNetPlusPlus(encoder_name, in_channels, num_classes)
    else:
        return UNet(encoder_name, in_channels, num_classes)


if __name__ == "__main__":
    # Test the model
    model = create_unet_model(encoder_name="resnet34", unet_plus_plus=True)
    x = torch.randn(1, 3, 768, 768)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Encoder: {model.encoder_name}")
