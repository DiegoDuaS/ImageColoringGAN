"""
Discriminador PatchGAN para Colorización de Imágenes
Proyecto Final Deep Learning

Arquitectura del Discriminador
- PatchGAN ajustado para imágenes 128x128
- Output: 14x14 patches (cada uno evalúa ~34x34 píxeles)
- Usa BCEWithLogitsLoss con one-sided label smoothing

Uso:
    from src.models.discriminator import PatchGANDiscriminator, DiscriminatorLoss
    
    discriminator = PatchGANDiscriminator(input_channels=3, features=64)
    criterion = DiscriminatorLoss(real_label=0.9, fake_label=0.0)
"""

import torch
import torch.nn as nn

class DiscriminatorBlock(nn.Module):
    """
    Bloque convolucional básico del discriminador PatchGAN.
    
    Arquitectura:
        Conv2d(kernel=4, stride, padding=1) → [BatchNorm] → LeakyReLU(0.2)
    
    Args:
        in_channels (int): Número de canales de entrada
        out_channels (int): Número de canales de salida
        stride (int): Stride de la convolución (1 o 2). Default: 2
        normalize (bool): Si aplicar BatchNorm. Default: True
    
    Shape:
        - Input: (B, in_channels, H, W)
        - Output: (B, out_channels, H//stride, W//stride)
    """
    
    def __init__(self, in_channels, out_channels, stride=2, normalize=True):
        super(DiscriminatorBlock, self).__init__()
        
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=False if normalize else True
            )
        ]
        
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class PatchGANDiscriminator(nn.Module):
    """
    Discriminador PatchGAN para imágenes 128x128.
    
    Evalúa la imagen en patches locales en vez de globalmente.
    Cada elemento del output evalúa si un patch de la imagen es real o falso.
    
    Arquitectura (para 128x128):
        Input (LAB) → 128x128 → 64x64 → 32x32 → 16x16 → 15x15 → 14x14 (output)
        Receptive field por patch: ~34x34 píxeles
    
    Args:
        input_channels (int): Número de canales de entrada. Default: 3 (LAB)
        features (int): Número de features base. Default: 64
    
    Shape:
        - Input: (B, 3, 128, 128) - Imagen LAB
          * Canal 0: L (luminancia)
          * Canal 1: a (verde-rojo)
          * Canal 2: b (azul-amarillo)
        - Output: (B, 1, 14, 14) - Matriz de logits (sin sigmoid)
          * Cada elemento evalúa un patch de ~34x34 píxeles
    
    Example:
        discriminator = PatchGANDiscriminator(input_channels=3, features=64)
        lab_image = torch.randn(4, 3, 128, 128)  # Batch de 4 imágenes LAB
        output = discriminator(lab_image)
        print(output.shape)  # torch.Size([4, 1, 14, 14])
    
    Note:
        - No usar sigmoid en el output (BCEWithLogitsLoss lo hace internamente)
        - Primera capa sin BatchNorm (estándar en discriminadores)
        - Última capa sin activación
    """
    
    def __init__(self, input_channels=3, features=64):
        super(PatchGANDiscriminator, self).__init__()
        
        # Layer 1: 128x128 → 64x64 (sin BatchNorm en primera capa)
        self.layer1 = DiscriminatorBlock(
            input_channels, features,
            stride=2, normalize=False
        )
        
        # Layer 2: 64x64 → 32x32
        self.layer2 = DiscriminatorBlock(
            features, features * 2,
            stride=2, normalize=True
        )
        
        # Layer 3: 32x32 → 16x16
        self.layer3 = DiscriminatorBlock(
            features * 2, features * 4,
            stride=2, normalize=True
        )
        
        # Layer 4: 16x16 → 15x15 (stride=1 para control de receptive field)
        self.layer4 = DiscriminatorBlock(
            features * 4, features * 8,
            stride=1, normalize=True
        )
        
        # Output layer: 15x15 → 14x14 (sin BatchNorm, sin activación)
        self.output_layer = nn.Conv2d(
            features * 8, 1,
            kernel_size=4,
            stride=1,
            padding=1
        )
    
    def forward(self, x):
        """
        Forward pass del discriminador.
        
        Args:
            x (torch.Tensor): Imagen LAB (B, 3, 128, 128)
        
        Returns:
            torch.Tensor: Logits (B, 1, 14, 14)
        """
        x = self.layer1(x)          # (B, 64, 64, 64)
        x = self.layer2(x)          # (B, 128, 32, 32)
        x = self.layer3(x)          # (B, 256, 16, 16)
        x = self.layer4(x)          # (B, 512, 15, 15)
        x = self.output_layer(x)    # (B, 1, 14, 14)
        return x
    
    def get_num_params(self):
        """Retorna el número total de parámetros del modelo."""
        return sum(p.numel() for p in self.parameters())


class DiscriminatorLoss(nn.Module):
    """
    Función de pérdida del discriminador con label smoothing.
    
    Combina las pérdidas de clasificar imágenes reales y falsas:
        L_D = (L_real + L_fake) / 2
    
    donde:
        - L_real: pérdida al clasificar imágenes reales
        - L_fake: pérdida al clasificar imágenes falsas (generadas)
    
    Usa one-sided label smoothing (recomendado en "Improved Techniques for Training GANs"):
        - Reales: 0.9 (suavizado)
        - Falsos: 0.0 (sin suavizar)
    
    Args:
        real_label (float): Label para imágenes reales. Default: 0.9
        fake_label (float): Label para imágenes falsas. Default: 0.0
    
    Returns:
        tuple: (total_loss, real_loss, fake_loss, real_accuracy, fake_accuracy)
    
    Example:
        criterion = DiscriminatorLoss(real_label=0.9, fake_label=0.0)
        real_pred = discriminator(real_images)
        fake_pred = discriminator(fake_images)
        loss, real_loss, fake_loss, real_acc, fake_acc = criterion(real_pred, fake_pred)
    """
    
    def __init__(self, real_label=0.9, fake_label=0.0):
        super(DiscriminatorLoss, self).__init__()
        self.real_label = real_label
        self.fake_label = fake_label
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, real_pred, fake_pred):
        """
        Calcula la pérdida del discriminador.
        
        Args:
            real_pred (torch.Tensor): Predicción en imágenes reales (B, 1, 14, 14)
            fake_pred (torch.Tensor): Predicción en imágenes falsas (B, 1, 14, 14)
        
        Returns:
            tuple:
                - total_loss (torch.Tensor): Pérdida total del discriminador
                - real_loss (torch.Tensor): Pérdida en imágenes reales
                - fake_loss (torch.Tensor): Pérdida en imágenes falsas
                - real_accuracy (float): % de reales correctamente clasificados
                - fake_accuracy (float): % de falsos correctamente clasificados
        """
        # Crear labels con smoothing
        real_labels = torch.full_like(real_pred, self.real_label)
        fake_labels = torch.full_like(fake_pred, self.fake_label)
        
        # Calcular pérdidas individuales
        real_loss = self.criterion(real_pred, real_labels)
        fake_loss = self.criterion(fake_pred, fake_labels)
        
        # Pérdida total (promedio)
        total_loss = (real_loss + fake_loss) / 2
        
        # Calcular accuracy para monitoreo
        with torch.no_grad():
            real_pred_sigmoid = torch.sigmoid(real_pred)
            fake_pred_sigmoid = torch.sigmoid(fake_pred)
            
            real_accuracy = (real_pred_sigmoid > 0.5).float().mean().item()
            fake_accuracy = (fake_pred_sigmoid < 0.5).float().mean().item()
        
        return total_loss, real_loss, fake_loss, real_accuracy, fake_accuracy


def init_discriminator_weights(discriminator):
    """
    Inicializa los pesos del discriminador siguiendo la estrategia DCGAN.
    
    - Convoluciones: Normal(μ=0, σ=0.02)
    - BatchNorm weights: Normal(μ=1, σ=0.02)
    - BatchNorm bias: 0
    
    Args:
        discriminator (nn.Module): Modelo discriminador
    
    Example:
        discriminator = PatchGANDiscriminator()
        init_discriminator_weights(discriminator)
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif 'BatchNorm' in classname:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
    
    discriminator.apply(init_func)


# Configuración recomendada
RECOMMENDED_CONFIG = {
    'input_channels': 3,
    'features': 64,
    'learning_rate': 2e-4,
    'betas': (0.5, 0.999),
    'real_label': 0.9,
    'fake_label': 0.0,
    'gradient_clip': 1.0
}


if __name__ == "__main__":
    # Test del módulo
    print("PRUEBA DEL DISCRIMINADOR PATCHGAN")
    
    # Crear discriminador
    discriminator = PatchGANDiscriminator(input_channels=3, features=64)
    
    # Inicializar pesos
    init_discriminator_weights(discriminator)
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 128, 128)
    
    with torch.no_grad():
        output = discriminator(test_input)
    
    print(f"\nForward pass exitoso")
    print(f"   Input shape:  {tuple(test_input.shape)}")
    print(f"   Output shape: {tuple(output.shape)}")
    print(f"   Parámetros:   {discriminator.get_num_params():,}")
    
    # Test función de pérdida
    criterion = DiscriminatorLoss(real_label=0.9, fake_label=0.0)
    
    real_pred = torch.randn(batch_size, 1, 14, 14)
    fake_pred = torch.randn(batch_size, 1, 14, 14)
    
    loss, real_loss, fake_loss, real_acc, fake_acc = criterion(real_pred, fake_pred)
    
    print(f"\nFunción de pérdida funcionando")
    print(f"   Loss total: {loss.item():.4f}")
    print(f"   Loss real:  {real_loss.item():.4f}")
    print(f"   Loss fake:  {fake_loss.item():.4f}")
    print(f"   Accuracy real: {real_acc*100:.1f}%")
    print(f"   Accuracy fake: {fake_acc*100:.1f}%")
    
    print("MÓDULO LISTO PARA USAR")
    