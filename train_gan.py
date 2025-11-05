"""
Loop de Entrenamiento y Optimización - GAN Colorización
========================================================
Script completo para entrenar la GAN de colorización con:
- Ciclo completo de entrenamiento
- Optimizadores (Adam con schedulers)
- Checkpointing y guardado de modelos
- Monitoreo de pérdidas y convergencia
- Experimentación con learning rates
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from tqdm import tqdm

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

class TrainingConfig:
    """Configuración centralizada del entrenamiento"""
    
    def __init__(self):
        # Directorios
        self.data_dir = "./ImagesProcessed/ImagesProcessed/color"
        self.checkpoint_dir = "checkpoints"
        self.samples_dir = "training_samples"
        self.logs_dir = "logs"
        
        # Hiperparámetros de entrenamiento
        self.num_epochs = 100
        self.batch_size = 16
        self.num_workers = 4
        
        # Optimizadores
        self.lr_generator = 2e-4
        self.lr_discriminator = 2e-4
        self.beta1 = 0.5
        self.beta2 = 0.999
        
        # Schedulers
        self.use_scheduler = True
        self.scheduler_type = "step"  # "step", "cosine", "plateau"
        self.step_size = 30
        self.gamma = 0.5
        
        # Pérdidas
        self.lambda_l1 = 100.0  # Peso de la pérdida L1
        
        # Checkpointing
        self.save_every = 5  # Guardar cada N épocas
        self.keep_last_n = 3  # Mantener últimos N checkpoints
        
        # Validación y monitoreo
        self.validate_every = 1
        self.log_every = 10  # Log cada N batches
        self.sample_every = 1  # Guardar samples cada N épocas
        self.num_samples = 8
        
        # Dataset
        self.num_classes = 8
        self.image_size = 128
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def to_dict(self) -> Dict:
        """Convierte la configuración a diccionario"""
        return {k: str(v) if isinstance(v, (Path, torch.device)) else v 
                for k, v in self.__dict__.items()}
    
    def save(self, path: str):
        """Guarda la configuración en JSON"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)


# =============================================================================
# MAPEO DE CATEGORÍAS
# =============================================================================

CATEGORY_MAP = {
    1: "airplane",
    2: "car",
    3: "cat",
    4: "dog",
    5: "flower",
    6: "fruit",
    7: "motorbike",
    8: "person"
}


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================

class CheckpointManager:
    """Gestiona guardado y carga de checkpoints"""
    
    def __init__(self, checkpoint_dir: str, keep_last_n: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        
    def save_checkpoint(
        self,
        epoch: int,
        generator: nn.Module,
        discriminator: nn.Module,
        optim_G: optim.Optimizer,
        optim_D: optim.Optimizer,
        scheduler_G: Optional[optim.lr_scheduler._LRScheduler] = None,
        scheduler_D: Optional[optim.lr_scheduler._LRScheduler] = None,
        metrics: Optional[Dict] = None,
        is_best: bool = False
    ):
        """
        Guarda un checkpoint completo del entrenamiento
        
        Args:
            epoch: Época actual
            generator: Modelo generador
            discriminator: Modelo discriminador
            optim_G: Optimizador del generador
            optim_D: Optimizador del discriminador
            scheduler_G: Scheduler del generador (opcional)
            scheduler_D: Scheduler del discriminador (opcional)
            metrics: Métricas a guardar (opcional)
            is_best: Si es el mejor modelo hasta ahora
        """
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optim_G_state_dict': optim_G.state_dict(),
            'optim_D_state_dict': optim_D.state_dict(),
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat()
        }
        
        if scheduler_G is not None:
            checkpoint['scheduler_G_state_dict'] = scheduler_G.state_dict()
        if scheduler_D is not None:
            checkpoint['scheduler_D_state_dict'] = scheduler_D.state_dict()
        
        # Guardar checkpoint regular
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint guardado: {checkpoint_path}")
        
        # Guardar como último checkpoint
        latest_path = self.checkpoint_dir / "checkpoint_latest.pth"
        torch.save(checkpoint, latest_path)
        
        # Si es el mejor, guardarlo también
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pth"
            torch.save(checkpoint, best_path)
            print(f"★ Mejor modelo guardado: {best_path}")
        
        # Limpiar checkpoints antiguos
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Elimina checkpoints antiguos, manteniendo solo los últimos N"""
        checkpoints = sorted(
            [f for f in self.checkpoint_dir.glob("checkpoint_epoch_*.pth")],
            key=lambda x: x.stat().st_mtime
        )
        
        if len(checkpoints) > self.keep_last_n:
            for old_checkpoint in checkpoints[:-self.keep_last_n]:
                old_checkpoint.unlink()
                print(f"✗ Checkpoint antiguo eliminado: {old_checkpoint.name}")
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        generator: nn.Module,
        discriminator: nn.Module,
        optim_G: optim.Optimizer,
        optim_D: optim.Optimizer,
        scheduler_G: Optional[optim.lr_scheduler._LRScheduler] = None,
        scheduler_D: Optional[optim.lr_scheduler._LRScheduler] = None
    ) -> Dict:
        """
        Carga un checkpoint
        
        Returns:
            Dict con información del checkpoint cargado
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optim_G.load_state_dict(checkpoint['optim_G_state_dict'])
        optim_D.load_state_dict(checkpoint['optim_D_state_dict'])
        
        if scheduler_G is not None and 'scheduler_G_state_dict' in checkpoint:
            scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        if scheduler_D is not None and 'scheduler_D_state_dict' in checkpoint:
            scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
        
        print(f"✓ Checkpoint cargado desde época {checkpoint['epoch']}")
        return checkpoint
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Encuentra el último checkpoint guardado"""
        latest_path = self.checkpoint_dir / "checkpoint_latest.pth"
        if latest_path.exists():
            return str(latest_path)
        return None


# =============================================================================
# MÉTRICAS Y MONITOREO
# =============================================================================

class MetricsTracker:
    """Rastrea y registra métricas del entrenamiento"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Historial de métricas
        self.history = {
            'train_loss_D': [],
            'train_loss_G': [],
            'train_loss_G_gan': [],
            'train_loss_G_l1': [],
            'val_loss_D': [],
            'val_loss_G': [],
            'learning_rates': {'G': [], 'D': []}
        }
        
        # Mejor pérdida para early stopping
        self.best_val_loss_G = float('inf')
        
    def update(
        self,
        epoch: int,
        metrics: Dict[str, float],
        phase: str = 'train'
    ):
        """
        Actualiza métricas
        
        Args:
            epoch: Época actual
            metrics: Diccionario con métricas
            phase: 'train' o 'val'
        """
        for key, value in metrics.items():
            full_key = f"{phase}_{key}"
            if full_key in self.history:
                self.history[full_key].append(value)
            
            # Log a TensorBoard
            self.writer.add_scalar(f"{phase}/{key}", value, epoch)
    
    def log_learning_rates(self, epoch: int, lr_G: float, lr_D: float):
        """Registra learning rates"""
        self.history['learning_rates']['G'].append(lr_G)
        self.history['learning_rates']['D'].append(lr_D)
        self.writer.add_scalar('learning_rate/generator', lr_G, epoch)
        self.writer.add_scalar('learning_rate/discriminator', lr_D, epoch)
    
    def is_best_model(self, val_loss_G: float) -> bool:
        """Determina si es el mejor modelo hasta ahora"""
        if val_loss_G < self.best_val_loss_G:
            self.best_val_loss_G = val_loss_G
            return True
        return False
    
    def plot_history(self, save_path: Optional[str] = None):
        """Genera gráficas de las métricas"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Pérdidas del discriminador
        if self.history['train_loss_D']:
            axes[0, 0].plot(self.history['train_loss_D'], label='Train')
            if self.history['val_loss_D']:
                axes[0, 0].plot(self.history['val_loss_D'], label='Val')
            axes[0, 0].set_title('Pérdida del Discriminador')
            axes[0, 0].set_xlabel('Época')
            axes[0, 0].set_ylabel('Pérdida')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Pérdidas del generador
        if self.history['train_loss_G']:
            axes[0, 1].plot(self.history['train_loss_G'], label='Train Total')
            if self.history['val_loss_G']:
                axes[0, 1].plot(self.history['val_loss_G'], label='Val Total')
            axes[0, 1].set_title('Pérdida del Generador')
            axes[0, 1].set_xlabel('Época')
            axes[0, 1].set_ylabel('Pérdida')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Componentes de pérdida del generador
        if self.history['train_loss_G_gan']:
            axes[1, 0].plot(self.history['train_loss_G_gan'], label='GAN Loss')
            axes[1, 0].plot(self.history['train_loss_G_l1'], label='L1 Loss')
            axes[1, 0].set_title('Componentes de Pérdida del Generador')
            axes[1, 0].set_xlabel('Época')
            axes[1, 0].set_ylabel('Pérdida')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning rates
        if self.history['learning_rates']['G']:
            axes[1, 1].plot(self.history['learning_rates']['G'], label='Generator')
            axes[1, 1].plot(self.history['learning_rates']['D'], label='Discriminator')
            axes[1, 1].set_title('Learning Rates')
            axes[1, 1].set_xlabel('Época')
            axes[1, 1].set_ylabel('LR')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Gráficas guardadas: {save_path}")
        else:
            plt.show()
        
        plt.close(fig)
    
    def save_history(self, path: str):
        """Guarda historial de métricas"""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def close(self):
        """Cierra el writer de TensorBoard"""
        self.writer.close()


# =============================================================================
# VISUALIZACIÓN DE SAMPLES
# =============================================================================

def save_colorization_samples(
    gray: torch.Tensor,
    fake_ab: torch.Tensor,
    real_ab: torch.Tensor,
    labels: torch.Tensor,
    epoch: int,
    save_dir: str,
    max_samples: int = 4
):
    """
    Guarda samples de colorización para monitoreo visual
    
    Args:
        gray: Imágenes en escala de grises [B, 1, H, W]
        fake_ab: Colores generados [B, 2, H, W]
        real_ab: Colores reales [B, 2, H, W]
        labels: Labels one-hot [B, num_classes]
        epoch: Época actual
        save_dir: Directorio de guardado
        max_samples: Número máximo de samples a guardar
    """
    os.makedirs(save_dir, exist_ok=True)
    num_samples = min(max_samples, gray.size(0))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)
    
    for i in range(num_samples):
        # Desnormalizar
        L = (gray[i, 0].cpu().numpy() + 1.0) * 50.0
        real_ab_np = real_ab[i].cpu().numpy() * 110.0
        fake_ab_np = fake_ab[i].detach().cpu().numpy() * 110.0
        
        # Convertir LAB → RGB
        def lab_to_rgb(L, ab):
            lab = np.zeros((L.shape[0], L.shape[1], 3))
            lab[:, :, 0] = L
            lab[:, :, 1:] = ab.transpose(1, 2, 0)
            rgb = np.clip(lab2rgb(lab), 0, 1)
            return rgb
        
        gray_img = np.repeat((L / 100.0)[..., None], 3, axis=2)
        real_rgb = lab_to_rgb(L, real_ab_np)
        fake_rgb = lab_to_rgb(L, fake_ab_np)
        
        # Mostrar imágenes
        imgs = [gray_img, real_rgb, fake_rgb]
        titles = ["Entrada (Grayscale)", "Ground Truth", "Generado"]
        
        for j, ax in enumerate(axes[i]):
            ax.imshow(imgs[j])
            ax.axis("off")
            ax.set_title(titles[j], fontsize=12, fontweight='bold')
        
        # Añadir etiqueta de categoría
        label_idx = labels[i].argmax().item() + 1
        category = CATEGORY_MAP.get(label_idx, f"Class {label_idx}")
        axes[i, 0].set_ylabel(
            category,
            rotation=0,
            labelpad=50,
            fontsize=11,
            va='center',
            fontweight='bold'
        )
    
    plt.suptitle(f"Época {epoch}", fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"samples_epoch_{epoch:04d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Samples guardados: {save_path}")


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

class GANTrainer:
    """Clase principal para entrenar la GAN de colorización"""
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        config: TrainingConfig
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        
        # Mover modelos al device
        self.generator.to(config.device)
        self.discriminator.to(config.device)
        
        # Criterios de pérdida
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()
        
        # Optimizadores
        self.optim_G = optim.Adam(
            self.generator.parameters(),
            lr=config.lr_generator,
            betas=(config.beta1, config.beta2)
        )
        self.optim_D = optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr_discriminator,
            betas=(config.beta1, config.beta2)
        )
        
        # Schedulers
        self.scheduler_G = None
        self.scheduler_D = None
        if config.use_scheduler:
            self.scheduler_G = self._create_scheduler(self.optim_G, config)
            self.scheduler_D = self._create_scheduler(self.optim_D, config)
        
        # Managers
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint_dir,
            keep_last_n=config.keep_last_n
        )
        self.metrics_tracker = MetricsTracker(config.logs_dir)
        
        # Estado del entrenamiento
        self.start_epoch = 0
        self.global_step = 0
    
    def _create_scheduler(
        self,
        optimizer: optim.Optimizer,
        config: TrainingConfig
    ) -> optim.lr_scheduler._LRScheduler:
        """Crea un learning rate scheduler"""
        if config.scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.step_size,
                gamma=config.gamma
            )
        elif config.scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.num_epochs
            )
        elif config.scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            raise ValueError(f"Scheduler type '{config.scheduler_type}' no reconocido")
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None):
        """
        Carga un checkpoint para reanudar entrenamiento
        
        Args:
            checkpoint_path: Ruta al checkpoint. Si es None, busca el último.
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_manager.find_latest_checkpoint()
            if checkpoint_path is None:
                print("No se encontró checkpoint previo. Iniciando desde cero.")
                return
        
        checkpoint = self.checkpoint_manager.load_checkpoint(
            checkpoint_path,
            self.generator,
            self.discriminator,
            self.optim_G,
            self.optim_D,
            self.scheduler_G,
            self.scheduler_D
        )
        
        self.start_epoch = checkpoint['epoch'] + 1
        print(f"Reanudando entrenamiento desde época {self.start_epoch}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Entrena una época completa
        
        Returns:
            Dict con métricas promedio de la época
        """
        self.generator.train()
        self.discriminator.train()
        
        metrics = {
            'loss_D': 0.0,
            'loss_G': 0.0,
            'loss_G_gan': 0.0,
            'loss_G_l1': 0.0
        }
        
        pbar = tqdm(train_loader, desc=f"Época {epoch}/{self.config.num_epochs}")
        
        for batch_idx, (gray, ab_color, labels) in enumerate(pbar):
            # Mover datos al device
            gray = gray.to(self.config.device)
            ab_color = ab_color.to(self.config.device)
            labels = labels.to(self.config.device)
            
            batch_size = gray.size(0)
            
            # ==========================================
            # 1. Entrenar Discriminador
            # ==========================================
            self.optim_D.zero_grad()
            
            # Forward del generador (sin gradientes)
            with torch.no_grad():
                fake_ab = self.generator(gray, labels)
            
            # Discriminador sobre imágenes reales
            real_input = torch.cat([gray, ab_color], dim=1)
            real_pred = self.discriminator(real_input)
            loss_D_real = self.criterion_gan(
                real_pred,
                torch.ones_like(real_pred)
            )
            
            # Discriminador sobre imágenes falsas
            fake_input = torch.cat([gray, fake_ab], dim=1)
            fake_pred = self.discriminator(fake_input)
            loss_D_fake = self.criterion_gan(
                fake_pred,
                torch.zeros_like(fake_pred)
            )
            
            # Pérdida total del discriminador
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            self.optim_D.step()
            
            # ==========================================
            # 2. Entrenar Generador
            # ==========================================
            self.optim_G.zero_grad()
            
            # Forward del generador
            fake_ab = self.generator(gray, labels)
            
            # Pérdida GAN
            fake_input = torch.cat([gray, fake_ab], dim=1)
            fake_pred = self.discriminator(fake_input)
            loss_G_gan = self.criterion_gan(
                fake_pred,
                torch.ones_like(fake_pred)
            )
            
            # Pérdida L1
            loss_G_l1 = self.criterion_l1(fake_ab, ab_color) * self.config.lambda_l1
            
            # Pérdida total del generador
            loss_G = loss_G_gan + loss_G_l1
            loss_G.backward()
            self.optim_G.step()
            
            # ==========================================
            # 3. Actualizar métricas
            # ==========================================
            metrics['loss_D'] += loss_D.item()
            metrics['loss_G'] += loss_G.item()
            metrics['loss_G_gan'] += loss_G_gan.item()
            metrics['loss_G_l1'] += loss_G_l1.item()
            
            # Actualizar barra de progreso
            pbar.set_postfix({
                'D': f"{loss_D.item():.4f}",
                'G': f"{loss_G.item():.4f}",
                'G_gan': f"{loss_G_gan.item():.4f}",
                'G_l1': f"{loss_G_l1.item():.4f}"
            })
            
            # Log periódico a TensorBoard
            if batch_idx % self.config.log_every == 0:
                self.metrics_tracker.writer.add_scalar(
                    'batch/loss_D',
                    loss_D.item(),
                    self.global_step
                )
                self.metrics_tracker.writer.add_scalar(
                    'batch/loss_G',
                    loss_G.item(),
                    self.global_step
                )
            
            self.global_step += 1
        
        # Promediar métricas
        num_batches = len(train_loader)
        metrics = {k: v / num_batches for k, v in metrics.items()}
        
        return metrics
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Evalúa el modelo en el conjunto de validación
        
        Returns:
            Dict con métricas de validación
        """
        self.generator.eval()
        self.discriminator.eval()
        
        metrics = {
            'loss_D': 0.0,
            'loss_G': 0.0,
            'loss_G_gan': 0.0,
            'loss_G_l1': 0.0
        }
        
        # Para guardar samples
        save_samples = True
        
        for batch_idx, (gray, ab_color, labels) in enumerate(val_loader):
            gray = gray.to(self.config.device)
            ab_color = ab_color.to(self.config.device)
            labels = labels.to(self.config.device)
            
            # Forward del generador
            fake_ab = self.generator(gray, labels)
            
            # Pérdida del discriminador
            real_input = torch.cat([gray, ab_color], dim=1)
            fake_input = torch.cat([gray, fake_ab], dim=1)
            
            real_pred = self.discriminator(real_input)
            fake_pred = self.discriminator(fake_input)
            
            loss_D_real = self.criterion_gan(real_pred, torch.ones_like(real_pred))
            loss_D_fake = self.criterion_gan(fake_pred, torch.zeros_like(fake_pred))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            
            # Pérdida del generador
            loss_G_gan = self.criterion_gan(fake_pred, torch.ones_like(fake_pred))
            loss_G_l1 = self.criterion_l1(fake_ab, ab_color) * self.config.lambda_l1
            loss_G = loss_G_gan + loss_G_l1
            
            # Acumular métricas
            metrics['loss_D'] += loss_D.item()
            metrics['loss_G'] += loss_G.item()
            metrics['loss_G_gan'] += loss_G_gan.item()
            metrics['loss_G_l1'] += loss_G_l1.item()
            
            # Guardar samples (solo del primer batch)
            if save_samples and epoch % self.config.sample_every == 0:
                save_colorization_samples(
                    gray,
                    fake_ab,
                    ab_color,
                    labels,
                    epoch,
                    self.config.samples_dir,
                    max_samples=self.config.num_samples
                )
                save_samples = False
        
        # Promediar métricas
        num_batches = len(val_loader)
        metrics = {k: v / num_batches for k, v in metrics.items()}
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ):
        """
        Ciclo completo de entrenamiento
        
        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación (opcional)
        """
        print("\n" + "="*70)
        print("INICIANDO ENTRENAMIENTO DE GAN PARA COLORIZACIÓN")
        print("="*70)
        print(f"Device: {self.config.device}")
        print(f"Épocas: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate G: {self.config.lr_generator}")
        print(f"Learning rate D: {self.config.lr_discriminator}")
        print(f"Scheduler: {self.config.scheduler_type if self.config.use_scheduler else 'None'}")
        print("="*70 + "\n")
        
        # Guardar configuración
        self.config.save(os.path.join(self.config.logs_dir, "config.json"))
        
        # Intentar cargar checkpoint previo
        self.load_checkpoint()
        
        # Loop de entrenamiento
        for epoch in range(self.start_epoch, self.config.num_epochs):
            epoch_start_time = time.time()
            
            # ==========================================
            # ENTRENAMIENTO
            # ==========================================
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Log de métricas
            self.metrics_tracker.update(epoch, train_metrics, phase='train')
            
            # ==========================================
            # VALIDACIÓN
            # ==========================================
            if val_loader is not None and epoch % self.config.validate_every == 0:
                val_metrics = self.validate(val_loader, epoch)
                self.metrics_tracker.update(epoch, val_metrics, phase='val')
                
                # Verificar si es el mejor modelo
                is_best = self.metrics_tracker.is_best_model(val_metrics['loss_G'])
            else:
                val_metrics = None
                is_best = False
            
            # ==========================================
            # LEARNING RATE SCHEDULING
            # ==========================================
            if self.scheduler_G is not None:
                if self.config.scheduler_type == "plateau" and val_metrics is not None:
                    self.scheduler_G.step(val_metrics['loss_G'])
                    self.scheduler_D.step(val_metrics['loss_D'])
                else:
                    self.scheduler_G.step()
                    self.scheduler_D.step()
            
            # Log learning rates
            current_lr_G = self.optim_G.param_groups[0]['lr']
            current_lr_D = self.optim_D.param_groups[0]['lr']
            self.metrics_tracker.log_learning_rates(epoch, current_lr_G, current_lr_D)
            
            # ==========================================
            # CHECKPOINTING
            # ==========================================
            if (epoch + 1) % self.config.save_every == 0:
                all_metrics = train_metrics.copy()
                if val_metrics:
                    all_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    generator=self.generator,
                    discriminator=self.discriminator,
                    optim_G=self.optim_G,
                    optim_D=self.optim_D,
                    scheduler_G=self.scheduler_G,
                    scheduler_D=self.scheduler_D,
                    metrics=all_metrics,
                    is_best=is_best
                )
            
            # ==========================================
            # RESUMEN DE ÉPOCA
            # ==========================================
            epoch_time = time.time() - epoch_start_time
            
            print(f"\n{'='*70}")
            print(f"Época {epoch}/{self.config.num_epochs - 1} completada en {epoch_time:.2f}s")
            print(f"{'='*70}")
            print(f"TRAIN | D: {train_metrics['loss_D']:.4f} | "
                  f"G: {train_metrics['loss_G']:.4f} | "
                  f"G_gan: {train_metrics['loss_G_gan']:.4f} | "
                  f"G_l1: {train_metrics['loss_G_l1']:.4f}")
            
            if val_metrics:
                print(f"VAL   | D: {val_metrics['loss_D']:.4f} | "
                      f"G: {val_metrics['loss_G']:.4f} | "
                      f"G_gan: {val_metrics['loss_G_gan']:.4f} | "
                      f"G_l1: {val_metrics['loss_G_l1']:.4f}")
                if is_best:
                    print("★ ¡NUEVO MEJOR MODELO! ★")
            
            print(f"LR    | G: {current_lr_G:.2e} | D: {current_lr_D:.2e}")
            print(f"{'='*70}\n")
        
        # ==========================================
        # FIN DEL ENTRENAMIENTO
        # ==========================================
        print("\n" + "="*70)
        print("ENTRENAMIENTO COMPLETADO")
        print("="*70)
        
        # Guardar historial y gráficas finales
        history_path = os.path.join(self.config.logs_dir, "training_history.json")
        self.metrics_tracker.save_history(history_path)
        print(f"✓ Historial guardado: {history_path}")
        
        plots_path = os.path.join(self.config.logs_dir, "training_plots.png")
        self.metrics_tracker.plot_history(plots_path)
        
        self.metrics_tracker.close()
        print("\n¡Entrenamiento finalizado exitosamente! 🎉\n")


# =============================================================================
# FUNCIÓN PRINCIPAL
# =========================================================================

def main():
    """Función principal para ejecutar el entrenamiento"""
    
    # Importar modelos (asumiendo que están disponibles)
    try:
        from Modelo.generador import ColorGenerator
        from Modelo.discriminador import (
            PatchGANDiscriminator,
            init_discriminator_weights,
            RECOMMENDED_CONFIG
        )
        from image_driver import ColorizationDataset
    except ImportError as e:
        print(f"Error al importar módulos: {e}")
        print("Asegúrate de que los archivos estén en las rutas correctas.")
        return
    
    # ==========================================
    # CONFIGURACIÓN
    # ==========================================
    config = TrainingConfig()
    
    # Opcional: modificar configuración desde aquí
    # config.num_epochs = 50
    # config.batch_size = 32
    # config.lr_generator = 1e-4
    
    print("Configuración cargada:")
    print(json.dumps(config.to_dict(), indent=2))
    
    # ==========================================
    # DATASET Y DATALOADERS
    # ==========================================
    print("\nCargando datasets...")
    
    # Dataset completo
    full_dataset = ColorizationDataset(
        config.data_dir,
        num_classes=config.num_classes
    )
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size]
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # ==========================================
    # MODELOS
    # ==========================================
    print("\nInicializando modelos...")
    
    # Generador
    generator = ColorGenerator()
    num_params_G = sum(p.numel() for p in generator.parameters())
    print(f"✓ Generador creado | Parámetros: {num_params_G:,}")
    
    # Discriminador
    discriminator = PatchGANDiscriminator(
        input_channels=3,  # L + AB
        features=64
    )
    init_discriminator_weights(discriminator)
    num_params_D = discriminator.get_num_params()
    print(f"✓ Discriminador creado | Parámetros: {num_params_D:,}")
    
    # ==========================================
    # TRAINER
    # ==========================================
    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        config=config
    )
    
    # ==========================================
    # ENTRENAR
    # ==========================================
    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        print("\n\nEntrenamiento interrumpido por el usuario.")
        print("Guardando checkpoint final...")
        trainer.checkpoint_manager.save_checkpoint(
            epoch=trainer.start_epoch,
            generator=trainer.generator,
            discriminator=trainer.discriminator,
            optim_G=trainer.optim_G,
            optim_D=trainer.optim_D,
            scheduler_G=trainer.scheduler_G,
            scheduler_D=trainer.scheduler_D,
            metrics={},
            is_best=False
        )
        print("✓ Checkpoint guardado. Puedes reanudar el entrenamiento más tarde.")



if __name__ == '__main__':
    main()