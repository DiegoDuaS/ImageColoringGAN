"""
EXPLAINABLE AI PARA GAN DE COLORIZACIÓN - ÉPOCA 16 
================================================================
Implementa: Saliency Maps, Grad-CAM, Feature Attribution
Dependencias: torch, numpy, matplotlib, scikit-image
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from skimage.transform import resize as sk_resize


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

class XAIConfig:
    def __init__(self):
        self.checkpoint_path = "checkpoints/checkpoint_epoch_0016.pth"
        self.data_dir = "./ImagesProcessed/ImagesProcessed/color"
        self.output_dir = "Analysis/Explainable_AI"
        self.num_classes = 8
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Crear directorios
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        self.saliency_dir = Path(self.output_dir) / "saliency_maps"
        self.gradcam_dir = Path(self.output_dir) / "gradcam"
        self.attribution_dir = Path(self.output_dir) / "feature_attribution"
        for d in [self.saliency_dir, self.gradcam_dir, self.attribution_dir]:
            d.mkdir(exist_ok=True, parents=True)


# ============================================================================
# 1. SALIENCY MAPS
# ============================================================================

class SaliencyMapGenerator:
    """
    Genera mapas de saliencia mostrando qué píxeles de entrada
    influyen más en la salida de color
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def generate_vanilla_saliency(self, gray, labels):
        """
        Saliency Map Básico: Gradiente de la salida respecto a la entrada
        Retorna: dict con 'channel_a', 'channel_b', 'combined' en [0,1]
        """
        self.model.eval()

        gray_input = gray.clone().detach().to(self.device).requires_grad_(True)
        labels_input = labels.clone().detach().to(self.device)

        output = self.model(gray_input, labels_input)  # [B,2,H,W]

        saliency_a = self._compute_saliency_for_channel(gray_input, output, channel_idx=0)
        saliency_b = self._compute_saliency_for_channel(gray_input, output, channel_idx=1)

        return {
            'channel_a': saliency_a,
            'channel_b': saliency_b,
            'combined': (saliency_a + saliency_b) / 2.0
        }

    def _compute_saliency_for_channel(self, input_tensor, output, channel_idx: int):
        # Limpiar gradientes previos
        self.model.zero_grad()
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()

        # Sumar importancia del canal seleccionado
        channel_importance = output[:, channel_idx, :, :].sum()
        channel_importance.backward(retain_graph=True)

        # |grad| y normalización
        sal = input_tensor.grad.data.abs().detach().squeeze().cpu().numpy()
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        return sal

    def generate_smooth_saliency(self, gray, labels, n_samples=50, noise_level=0.1):
        """Promedia múltiples saliency maps con ruido gaussiano."""
        maps = []
        for _ in range(n_samples):
            noisy = (gray + torch.randn_like(gray) * noise_level).to(self.device)
            sal = self.generate_vanilla_saliency(noisy, labels)['combined']
            maps.append(sal)
        return np.mean(maps, axis=0)


# ============================================================================
# 2. GRAD-CAM
# ============================================================================

class GradCAM:
    """
    Grad-CAM adaptado a GANs de colorización (canal 'a' como escalar).
    No usa OpenCV: redimensiona con skimage y hace overlay con numpy.
    """
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activations = None
        self._hook_registered = False
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inputs, output):
            self.activations = output.detach()

        def bwd_hook(module, grad_input, grad_output):
            # grad_output es una tupla; [0] es el grad de la salida de la capa
            self.gradients = grad_output[0].detach()

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(fwd_hook)
                # register_full_backward_hook para PyTorch >= 1.8
                if hasattr(module, "register_full_backward_hook"):
                    module.register_full_backward_hook(bwd_hook)
                else:
                    module.register_backward_hook(bwd_hook)
                self._hook_registered = True
                break

    def list_layers(self):
        return [name for name, _ in self.model.named_modules()]

    def generate_gradcam(self, gray, labels):
        """
        Retorna heatmap [H, W] en [0,1] o None si no hay hook/capa
        """
        if not self._hook_registered:
            return None

        self.model.eval()
        gray_input = gray.clone().to(self.device).requires_grad_(True)
        labels_input = labels.clone().to(self.device)

        out = self.model(gray_input, labels_input)  # [B,2,H,W]
        score = out[:, 0, :, :].sum()  # canal 'a' como escalar de importancia

        self.model.zero_grad()
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return None

        # Ponderación por gradiente medio por canal
        # grad:[B,C,H',W'], act:[B,C,H',W']
        pooled_grad = torch.mean(self.gradients, dim=[0, 2, 3])  # [C]
        acts = self.activations.clone()

        for c in range(acts.shape[1]):
            acts[:, c, :, :] *= pooled_grad[c]

        heat = torch.mean(acts, dim=1).squeeze()     # [H', W']
        heat = F.relu(heat).detach().cpu().numpy()
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        return heat  # lo redimensionamos al volcarlo


# ============================================================================
# 3. FEATURE ATTRIBUTION
# ============================================================================

class FeatureAttribution:
    """
    Analiza cómo la categoría de condición (one-hot) influye en la colorización.
    """
    def __init__(self, model, device, num_classes=8):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.category_map = {
            0: "airplane", 1: "car", 2: "cat", 3: "dog",
            4: "flower", 5: "fruit", 6: "motorbike", 7: "person"
        }

    def analyze_category_influence(self, gray, true_label):
        self.model.eval()
        results = {}

        with torch.no_grad():
            gray_input = gray.to(self.device)
            for class_idx in range(self.num_classes):
                label = torch.zeros(1, self.num_classes, device=self.device)
                label[0, class_idx] = 1.0
                ab = self.model(gray_input, label)  # [1,2,H,W]
                results[self.category_map[class_idx]] = {
                    'ab': ab.cpu(),
                    'is_true': (class_idx == true_label.argmax().item())
                }
        return results

    def compute_category_impact(self, gray, true_label):
        results = self.analyze_category_influence(gray, true_label)
        true_cat = self.category_map[true_label.argmax().item()]
        true_ab = results[true_cat]['ab']

        impacts = {}
        for cname, data in results.items():
            if cname == true_cat:
                impacts[cname] = 0.0
            else:
                diff = torch.mean((true_ab - data['ab']) ** 2).item()
                impacts[cname] = diff
        return impacts, true_cat


# ============================================================================
# 4. VISUALIZADOR
# ============================================================================

class XAIVisualizer:
    def __init__(self, config):
        self.config = config
        self.category_map = {
            0: "airplane", 1: "car", 2: "cat", 3: "dog",
            4: "flower", 5: "fruit", 6: "motorbike", 7: "person"
        }

    @staticmethod
    def _lab_to_rgb(L, ab):
        """Convierte L (HxW) y ab (2xHxW) a RGB [0,1]."""
        lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.float32)
        lab[:, :, 0] = L
        lab[:, :, 1:] = ab.transpose(1, 2, 0)
        rgb = lab2rgb(lab)
        return np.clip(rgb, 0, 1)

    @staticmethod
    def _to_uint8(img):
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 1)
            img = (img * 255.0).astype(np.uint8)
        return img

    @staticmethod
    def _overlay_heatmap(img_rgb01, heatmap01, alpha=0.5, cmap='hot'):
        """
        Superpone heatmap (HxW, [0,1]) sobre img (HxWx3, [0,1]).
        """
        img = np.clip(img_rgb01, 0, 1).astype(np.float32)
        cmap_obj = plt.get_cmap(cmap)
        heat_rgb = cmap_obj(np.clip(heatmap01, 0, 1))[:, :, :3].astype(np.float32)  # [H,W,3]
        overlay = (1.0 - alpha) * img + alpha * heat_rgb
        return np.clip(overlay, 0, 1)

    def visualize_complete_analysis(
        self, gray_tensor, ab_real, ab_pred, label,
        saliency_maps, gradcam_map, category_analysis, sample_idx
    ):
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        gray_np = gray_tensor.squeeze().cpu().numpy()
        L = (gray_np + 1.0) * 50.0
        gray_img = np.repeat((L / 100.0)[..., None], 3, axis=2)  # RGB gris [0,1]

        real_rgb = self._lab_to_rgb(L, ab_real.squeeze().cpu().numpy() * 110.0)
        pred_rgb = self._lab_to_rgb(L, ab_pred.squeeze().cpu().numpy() * 110.0)

        cat_idx = label.argmax().item()
        cat_name = self.category_map[cat_idx]

        # Fila 1
        ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(gray_img); ax1.set_title('Entrada (Grayscale)', fontsize=12, fontweight='bold'); ax1.axis('off')
        ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(real_rgb); ax2.set_title('Ground Truth', fontsize=12, fontweight='bold'); ax2.axis('off')
        ax3 = fig.add_subplot(gs[0, 2]); ax3.imshow(pred_rgb); ax3.set_title('Generado por GAN', fontsize=12, fontweight='bold'); ax3.axis('off')
        ax4 = fig.add_subplot(gs[0, 3])
        diff = np.linalg.norm(real_rgb - pred_rgb, axis=2)  # L2 por píxel
        vmax = np.percentile(diff, 99)
        im4  = ax4.imshow(diff, vmin=0, vmax=max(vmax, 1e-3), cmap='inferno')
        ax4.set_title('Diferencia (Error)', fontsize=12, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046)
        # Fila 2: Saliency
        ax5 = fig.add_subplot(gs[1, 0]); im5 = ax5.imshow(saliency_maps['channel_a'], cmap='hot'); ax5.set_title('Saliency - Canal A\n(Verde-Rojo)', fontsize=11, fontweight='bold'); ax5.axis('off'); plt.colorbar(im5, ax=ax5, fraction=0.046)
        ax6 = fig.add_subplot(gs[1, 1]); im6 = ax6.imshow(saliency_maps['channel_b'], cmap='hot'); ax6.set_title('Saliency - Canal B\n(Azul-Amarillo)', fontsize=11, fontweight='bold'); ax6.axis('off'); plt.colorbar(im6, ax=ax6, fraction=0.046)
        ax7 = fig.add_subplot(gs[1, 2]); im7 = ax7.imshow(saliency_maps['combined'], cmap='hot'); ax7.set_title('Saliency - Combinado', fontsize=11, fontweight='bold'); ax7.axis('off'); plt.colorbar(im7, ax=ax7, fraction=0.046)
        ax8 = fig.add_subplot(gs[1, 3]); overlay_sal = self._overlay_heatmap(gray_img, saliency_maps['combined']); ax8.imshow(overlay_sal); ax8.set_title('Saliency Overlay', fontsize=11, fontweight='bold'); ax8.axis('off')

        # Fila 3: Grad-CAM + Feature Attribution
        ax9 = fig.add_subplot(gs[2, 0])
        ax10 = fig.add_subplot(gs[2, 1])

        if gradcam_map is not None:
            # Redimensionar Grad-CAM a tamaño de la imagen via skimage 
            H, W = gray_img.shape[:2]
            gcam_resized = sk_resize(gradcam_map, (H, W), preserve_range=True, anti_aliasing=True).astype(np.float32)
            im9 = ax9.imshow(gcam_resized, cmap='jet'); ax9.set_title('Grad-CAM\n(Regiones Importantes)', fontsize=11, fontweight='bold'); ax9.axis('off'); plt.colorbar(im9, ax=ax9, fraction=0.046)
            overlay_gcam = self._overlay_heatmap(gray_img, gcam_resized, cmap='jet')
            ax10.imshow(overlay_gcam); ax10.set_title('Grad-CAM Overlay', fontsize=11, fontweight='bold'); ax10.axis('off')
        else:
            ax9.text(0.5, 0.5, 'Grad-CAM no disponible', ha='center', va='center'); ax9.axis('off')
            ax10.axis('off')

        # Feature Attribution
        ax11 = fig.add_subplot(gs[2, 2:])
        impacts, true_cat = category_analysis
        sorted_cats = sorted(impacts.items(), key=lambda x: x[1], reverse=True)
        categories = [c for c, _ in sorted_cats]
        values = [v for _, v in sorted_cats]
        colors = ['green' if c == true_cat else 'skyblue' for c in categories]
        bars = ax11.barh(categories, values, color=colors, alpha=0.8, edgecolor='black')
        ax11.set_xlabel('Impacto en Colorización (Diferencia L2)', fontsize=11)
        ax11.set_title(f'Feature Attribution por Categoría\n(Verdadera: {true_cat})', fontsize=11, fontweight='bold')
        ax11.grid(True, axis='x', alpha=0.3)
        for b, v in zip(bars, values):
            if v > 0:
                ax11.text(v, b.get_y() + b.get_height()/2, f'{v:.4f}', va='center', ha='left', fontsize=9)

        plt.suptitle(f'Análisis Explainable AI - Muestra {sample_idx} (Categoría: {cat_name})',
                     fontsize=16, fontweight='bold', y=0.995)

        save_path = Path(self.config.output_dir) / f'complete_xai_analysis_sample_{sample_idx}.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Análisis XAI guardado: {save_path.name}")

    def generate_summary_report(self, all_impacts):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Promedio por categoría
        avg_imp = {}
        for d in all_impacts:
            for k, v in d.items():
                avg_imp.setdefault(k, []).append(v)
        avg_imp = {k: float(np.mean(v)) for k, v in avg_imp.items()}
        sorted_avg = sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)
        cats = [c for c, _ in sorted_avg]
        vals = [v for _, v in sorted_avg]

        ax1.barh(cats, vals, color='coral', alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Impacto Promedio', fontsize=12)
        ax1.set_title('Influencia Promedio de Cada Categoría\nen la Colorización',
                      fontsize=13, fontweight='bold')
        ax1.grid(True, axis='x', alpha=0.3)

        ax2.text(0.5, 0.5,
                 'Análisis de Categorías Completado\n\n'
                 f'Total de muestras analizadas: {len(all_impacts)}\n'
                 f'Categoría más distintiva: {cats[0] if cats else "N/A"}\n'
                 f'Categoría menos distintiva: {cats[-1] if cats else "N/A"}',
                 ha='center', va='center', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.axis('off')

        plt.suptitle('Resumen de Feature Attribution', fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = Path(self.config.output_dir) / 'feature_attribution_summary.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"\n✓ Resumen de Attribution guardado: {save_path.name}")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    print("\n" + "="*70)
    print("EXPLAINABLE AI - GAN DE COLORIZACIÓN (ÉPOCA 16)")
    print("="*70 + "\n")

    config = XAIConfig()

    # Importar tus módulos del proyecto
    try:
        from Modelo.generador import ColorGenerator
        from image_driver import ColorizationDataset
    except ImportError as e:
        print(f"Error al importar módulos: {e}")
        return

    # Modelo
    print("Cargando modelo...")
    generator = ColorGenerator()
    ckpt = torch.load(config.checkpoint_path, map_location=config.device)
    generator.load_state_dict(ckpt['generator_state_dict'])
    generator.to(config.device).eval()
    print("✓ Modelo cargado\n")

    # Dataset/Dataloader
    dataset = ColorizationDataset(config.data_dir, num_classes=config.num_classes)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Herramientas XAI
    saliency_gen = SaliencyMapGenerator(generator, config.device)

    # Ajusta 'up3' al nombre REAL de una capa conv del decoder
    gradcam = GradCAM(generator, target_layer='up3', device=config.device)
    # Si no estás seguro del nombre: descomenta para listar capas
    # print("\nCapas del generador:\n", gradcam.list_layers(), "\n")

    feature_attr = FeatureAttribution(generator, config.device, num_classes=config.num_classes)
    visualizer = XAIVisualizer(config)

    # Analizar N muestras
    num_samples = 5
    all_impacts = []
    print(f"Analizando {num_samples} muestras...\n")

    for idx, (gray, ab_real, label) in enumerate(dataloader):
        if idx >= num_samples:
            break

        print(f"Muestra {idx+1}/{num_samples}")

        gray = gray.to(config.device)
        ab_real = ab_real.to(config.device)
        label = label.to(config.device)

        with torch.no_grad():
            ab_pred = generator(gray, label)

        # 1) Saliency
        print("  - Generando Saliency Maps...")
        saliency_maps = saliency_gen.generate_vanilla_saliency(gray, label)

        # 2) Grad-CAM
        print("  - Generando Grad-CAM...")
        gradcam_map = None
        try:
            gradcam_map = gradcam.generate_gradcam(gray, label)
        except Exception as e:
            print(f"    (Grad-CAM no disponible: {e})")

        # 3) Feature Attribution
        print("  - Analizando Feature Attribution...")
        impacts, true_cat = feature_attr.compute_category_impact(gray, label)
        all_impacts.append(impacts)

        # 4) Visualización
        print("  - Generando visualización completa...")
        visualizer.visualize_complete_analysis(
            gray, ab_real, ab_pred, label,
            saliency_maps, gradcam_map,
            (impacts, true_cat),
            idx + 1
        )
        print()

    # Resumen
    print("Generando reporte resumen...")
    visualizer.generate_summary_report(all_impacts)

    print("\n" + "="*70)
    print("ANÁLISIS EXPLAINABLE AI COMPLETADO")
    print("="*70)
    print(f"\nResultados guardados en: {config.output_dir}\n")
    print("Archivos generados:")
    print("  • complete_xai_analysis_sample_X.png")
    print("  • feature_attribution_summary.png")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()