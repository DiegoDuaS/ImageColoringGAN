""" 
Estructura Generador: colorizador de imagenes GAN

Formato de imagenes es LAB en lugar de RGB, por lo que dada la imagen en blanco y negro
El modelo solo predice los canales AB.

Basado en estructuras convolucionales de Encoder-Decoder

- Encoder:
    Utiliza pesos pre-entrenados de VGG16 y aplica 4 capas de encoding y un bottleneck.
    En la ultima etapa de encoding y el bottleneck los pesos son descongelados, pues son las etapas mas cruciales
    de nuestro generador y podria llevar a mejores resultados entrenarlos tambien.
    
- Attention:
    Mecanismo simple de atencion con las caracteristicas de entrada

- Decoder:
    Basado en U-Net con 'skip' para mantener informacion semantica.
    Tambien posee bloque residual para resultados mas finos
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class GeneratorEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.enc1 = vgg[:4]
        self.enc2 = vgg[4:9]
        self.enc3 = vgg[9:16]
        self.enc4 = vgg[16:23]
        self.bottleneck = vgg[23:30]
        
        for p in self.parameters():
            p.requires_grad = False  # freeze VGG
            
        # Unfreze for critical stages
        for p in self.enc4.parameters():
            p.requires_grad = True
        for p in self.bottleneck.parameters():
            p.requires_grad = True
        
    def forward(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        bn = self.bottleneck(f4)
        return [f1, f2, f3, f4, bn]


## 2. Attention
class GeneratorAttention(nn.Module):
    def __init__(
            self,
            in_ch
        ):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//8, 1),
            nn.ReLU(),
            nn.Conv2d(in_ch//8, in_ch, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        w = self.att(x)
        return x * w

## Decoder (based on U-Net)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(
                channels    
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1    
            ),
            nn.BatchNorm2d(channels),
        )
    def forward(self, x):
        return F.relu(x + self.block(x))

class GeneratorDecoder(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels
        ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels + skip_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(
                out_channels    
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1    
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)

        return self.conv(x)

class ColorGenerator(nn.Module):
    def __init__(
            self
        ):
        super().__init__()
        self.input_adapter = nn.Conv2d(1 + 8, 3, 1)
        self.encoder = GeneratorEncoder()
        self.att = GeneratorAttention(512)
        
        self.up4 = GeneratorDecoder(512, 512, 256)
        self.res4 = ResidualBlock(256)
        
        self.up3 = GeneratorDecoder(256, 256, 128)
        self.res3 = ResidualBlock(128)
        
        self.up2 = GeneratorDecoder(128, 128, 64)
        self.res2 = ResidualBlock(64)
        
        self.up1 = GeneratorDecoder(64, 64, 32)
        self.res1 = ResidualBlock(32)
        
        self.final = nn.Conv2d(32, 2, 1)
        
        
    def forward(self, gray, class_vec):
        B, _, H, W = gray.shape # Shape
        class_map = class_vec.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)
        
        # Concat grayscale image and labels
        x = torch.cat([gray, class_map], dim=1)
        x = self.input_adapter(x) # adapter for VGG16 weights
        
        # Encoder
        f1, f2, f3, f4, bn = self.encoder(x)
        
        # Attention
        b = self.att(bn)
        
        # Decoder
        d4 = self.up4(b, f4)
        d4 = self.res4(d4)
        
        d3 = self.up3(d4, f3)
        d3 = self.res3(d3)
        
        d2 = self.up2(d3, f2)
        d2 = self.res2(d2)
        
        d1 = self.up1(d2, f1)
        d1 = self.res1(d1)
        
        # Final
        out = torch.tanh(self.final(d1))
        return out
    

if __name__ == "__main__":
    print("Hello world!")