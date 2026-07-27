# Image Coloring GAN

A conditional Generative Adversarial Network (cGAN) for automatic image colorization from grayscale images.

Instead of predicting an entire RGB image, the model operates in the CIELAB (LAB) color space. Since the grayscale image already contains the luminance (L) channel, the generator only learns to predict the a and b chromatic channels, reducing the complexity of the learning problem while preserving image brightness.

The generator is conditioned both on the grayscale image and a semantic category label, allowing the network to exploit contextual information during color prediction.

## 1. Architecture
The generator follows an Encoder–Attention–Decoder architecture inspired by U-Net.
```
Grayscale (L) + Class Label
            │
     Input Adapter (1×1 Conv)
            │
      VGG16 Encoder
            │
     Attention Module
            │
     U-Net Decoder
    + Skip Connections
    + Residual Blocks
            │
      Predicted AB Channels
```

### 1.1 Generator
The generator receives two inputs:

Grayscale image: (B, 1, 128, 128)
Semantic class vector: (B, 8)

The class vector is expanded spatially and concatenated with the grayscale image before entering the network.

### 1.2 Input Adapter
Since the encoder uses pretrained VGG16 weights, an initial 1×1 convolution converts the concatenated input

1 grayscale channel
+ 8 semantic channels

into the 3-channel representation expected by VGG16.

### 1.3 Encoder

The encoder is built from the convolutional feature extractor of VGG16 pretrained on ImageNet.

It is divided into four encoding stages followed by a bottleneck.
```
Encoder 1
Encoder 2
Encoder 3
Encoder 4
Bottleneck
```

To preserve pretrained representations while still allowing task adaptation:

Early layers remain frozen
The final encoder stage and bottleneck are fine-tuned

This strategy keeps generic low-level features learned from ImageNet while adapting higher-level semantic representations to the colorization task.

### 1.4 Attention Module

A lightweight channel-attention block is applied after the bottleneck.
It learns channel-wise importance weights through two 1×1 convolutions followed by a sigmoid activation.
```
Features
   │
1×1 Conv
   │
ReLU
   │
1×1 Conv
   │
Sigmoid
   │
Feature Reweighting
```

This allows the generator to emphasize informative feature maps before decoding.

### 1.5 Decoder

The decoder follows the U-Net design.

Each decoding stage consists of:

Bilinear upsampling
Skip connection with the corresponding encoder feature map
Two convolutional layers
Batch normalization
ReLU activation
Residual refinement block

Skip connections preserve fine spatial information that would otherwise be lost during downsampling, while residual blocks improve local color refinement.

The final layer outputs two channels corresponding to the predicted a and b components of the LAB color space.

A tanh activation constrains predictions to the normalized output range.

## 2. Data
### 2.1 Why LAB Instead of RGB?

Traditional RGB colorization requires predicting three image channels.

In LAB space:

L represents luminance (already available from the grayscale image)
a represents the green–red axis
b represents the blue–yellow axis

Therefore, the model only needs to learn color information instead of reconstructing image intensity, simplifying the learning problem and preserving structural details

### 2.2Conditional Colorization

Unlike unconditional colorization models, this generator receives a semantic label encoded as an 8-dimensional one-hot vector.

The available classes are:
```
Airplane
Car
Cat
Dog
Flower
Fruit
Motorbike
Person
```

Conditioning the generator provides semantic context that helps resolve color ambiguities (e.g., skies, vegetation, skin tones, vehicles, or flowers).

## 3. Training
The generator is trained adversarially against a PatchGAN discriminator using:

Adversarial Loss: Binary Cross Entropy with Logits
Reconstruction Loss: L1 Loss on the predicted LAB channels

The discriminator uses one-sided label smoothing and evaluates local image patches instead of the entire image, encouraging sharper and more realistic color predictions.

# 4. Output
Given: Grayscale image (L)
Generator: Predicts AB channels
Returns: LAB image → RGB conversion

the predicted AB channels are combined with the original L channel to reconstruct the final color image.

# 5. Future Improvements

Potential extensions include:

Self-attention or Transformer-based bottlenecks
Perceptual loss using VGG feature maps
Multi-scale discriminators
Progressive image resolutions
Diffusion-based refinement
Larger semantic conditioning or text-guided colorization

# 6. Credits
