# Model A design: custom CNN from scratch (Phase 2)

Model A is the "trained from scratch" arm of H1. Its learning rate (1e-3) is
large compared with half a bf16 step of its weights, so it should be far less
affected by bf16 weight freezing than Model C (fine-tuning at 1e-5). It is
built in Phase 3 as `build_model_a_scratch()` in `src/models.py`. This note
fixes the design so that the parameter counts can be checked against
`model.summary()` now.

## Architecture

Input 224 × 224 × 3 (RGB), 38 output classes. Four identical conv blocks,
then a small classifier head:

```
image 224x224x3
  -> Rescaling(1/255)
  -> [ Conv 3x3, 32  -> BatchNorm -> ReLU -> MaxPool 2x2 ]   block 1  -> 112x112x32
  -> [ Conv 3x3, 64  -> BatchNorm -> ReLU -> MaxPool 2x2 ]   block 2  ->  56x56x64
  -> [ Conv 3x3, 128 -> BatchNorm -> ReLU -> MaxPool 2x2 ]   block 3  ->  28x28x128
  -> [ Conv 3x3, 256 -> BatchNorm -> ReLU -> MaxPool 2x2 ]   block 4  ->  14x14x256
  -> GlobalAveragePooling2D                                           ->  256
  -> Dropout(0.3)                                                     ->  256
  -> Dense(38, softmax)                                               ->  38
```

All convolutions use stride 1, `padding="same"` and He-normal
initialisation. The whole network is wrapped in `src.precision.PrecisionModel`,
so the same architecture trains in fp32, bf16_rtn or bf16_sr.

## Output shapes and parameter counts (by hand)

Formulas:
- Conv2D: k × k × C_in × C_out + C_out (weights + one bias per filter)
- BatchNorm: 4 × C (γ and β are trainable; the moving mean and moving variance
  are not)
- Dense: n_in × n_out + n_out
- Rescaling, ReLU, MaxPool, GlobalAveragePooling and Dropout have no parameters

| # | Layer | Output shape | Hand count | Params |
|---|---|---|---|---:|
| 0 | Input | 224 × 224 × 3 | – | 0 |
| 1 | Rescaling 1/255 | 224 × 224 × 3 | – | 0 |
| 2 | block1 Conv 3×3, 32 | 224 × 224 × 32 | 3·3·3·32 + 32 = 864 + 32 | 896 |
| 3 | block1 BatchNorm | 224 × 224 × 32 | 4 · 32 | 128 |
| 4 | block1 ReLU | 224 × 224 × 32 | – | 0 |
| 5 | block1 MaxPool 2×2 | 112 × 112 × 32 | – | 0 |
| 6 | block2 Conv 3×3, 64 | 112 × 112 × 64 | 3·3·32·64 + 64 = 18,432 + 64 | 18,496 |
| 7 | block2 BatchNorm | 112 × 112 × 64 | 4 · 64 | 256 |
| 8 | block2 ReLU | 112 × 112 × 64 | – | 0 |
| 9 | block2 MaxPool 2×2 | 56 × 56 × 64 | – | 0 |
| 10 | block3 Conv 3×3, 128 | 56 × 56 × 128 | 3·3·64·128 + 128 = 73,728 + 128 | 73,856 |
| 11 | block3 BatchNorm | 56 × 56 × 128 | 4 · 128 | 512 |
| 12 | block3 ReLU | 56 × 56 × 128 | – | 0 |
| 13 | block3 MaxPool 2×2 | 28 × 28 × 128 | – | 0 |
| 14 | block4 Conv 3×3, 256 | 28 × 28 × 256 | 3·3·128·256 + 256 = 294,912 + 256 | 295,168 |
| 15 | block4 BatchNorm | 28 × 28 × 256 | 4 · 256 | 1,024 |
| 16 | block4 ReLU | 28 × 28 × 256 | – | 0 |
| 17 | block4 MaxPool 2×2 | 14 × 14 × 256 | – | 0 |
| 18 | GlobalAveragePooling2D | 256 | – | 0 |
| 19 | Dropout 0.3 | 256 | – | 0 |
| 20 | Dense 38, softmax | 38 | 256·38 + 38 = 9,728 + 38 | 9,766 |

Totals:

| | Hand count | `model.summary()` |
|---|---:|---:|
| Conv layers | 896 + 18,496 + 73,856 + 295,168 = 388,416 | |
| BatchNorm layers | 128 + 256 + 512 + 1,024 = 1,920 | |
| Dense head | 9,766 | |
| **Total** | 388,416 + 1,920 + 9,766 = **400,102** | 400,102 |
| **Trainable** | 400,102 − 960 = **399,142** | 399,142 |
| **Non-trainable** (moving mean + variance: 2 × (32+64+128+256)) | **960** | 960 |

Checked on 2026-11-25 with Keras 3.15.1: every per-layer count and all three
totals match `model.summary()`. The counts are the same at 160 px, because no
layer's parameters depend on the image size (see global average pooling below).

## Why each choice

- **224 × 224 × 3 input.** This is the same input as ResNet-50 in Models B and
  C, so all three models see identical images and differ only in architecture
  and training. If the Colab timing notebook moves the project to 160 px,
  Model A needs no change.
- **Rescaling 1/255 inside the model.** This maps pixels to [0, 1] with a fixed
  constant. Nothing is fitted, so it cannot leak validation or test
  information. The saved model also accepts raw pixels, which keeps the Gradio
  app simple. The data pipeline must therefore feed unnormalised 0–255 images
  to Model A.
- **3 × 3 kernels, stride 1, `same` padding.** Stacked 3 × 3 convolutions see
  as far as one larger kernel with fewer parameters: two 3 × 3 layers cover
  5 × 5 with 18·C² weights instead of 25·C². `same` padding keeps the size
  halving cleanly at each pool, 224 → 112 → 56 → 28 → 14.
- **Four blocks (the top of the 3–4 range).** Each block grows the receptive
  field. After 3 blocks a feature sees only 22 × 22 px; after 4 it sees
  46 × 46 px, about a fifth of the leaf image. That is enough to cover a whole
  lesion, ring or spot pattern. The 4th block costs 295k parameters but adds
  the 256 high-level features the head classifies from. A 5th block would
  shrink the map to 7 × 7 and add about 1.2M parameters, which is more than
  this budget allows.
- **Filters 32 → 64 → 128 → 256 (doubling).** Each pool divides the number of
  positions by 4, and doubling the filters multiplies the conv cost by 4, so
  blocks 2–4 cost the same (≈ 231M multiply-adds each). No block is a
  bottleneck. Block 1 starts small (32) because it runs at full 224 × 224
  resolution, where activations take the most memory.
- **BatchNorm between conv and ReLU.** BatchNorm keeps the pre-activation
  scale stable while every weight changes. That allows the relatively large
  lr of 1e-3 and makes training from scratch much less sensitive to
  initialisation. Conv → BN → ReLU is the order the spec asks for and the one
  in the original BatchNorm paper. *Considered and rejected:* BatchNorm's β
  makes the conv bias redundant, so `use_bias=False` would save
  32 + 64 + 128 + 256 = 480 parameters (0.12%). We keep the bias so that the
  counts follow the spec's formula.
- **ReLU.** ReLU is cheap and does not saturate for positive inputs, so
  gradients reach the early blocks. He-normal initialisation is the matching
  choice: it keeps the activation variance roughly constant through ReLU
  layers. Keras' default Glorot initialisation assumes linear or tanh units.
- **MaxPool 2 × 2.** Max pooling halves the resolution with no parameters. It
  keeps the strongest local response, which suits disease spots: whether a
  spot is present matters more than exactly where it is.
- **Global average pooling instead of Flatten.** Flatten of the 14 × 14 × 256
  map gives 50,176 features, and a Dense(38) on it would have
  50,176 · 38 + 38 = 1,906,726 parameters. That is 4.8× the rest of the model
  and would overfit badly. GAP averages each of the 256 feature maps to one
  number, so the head has 9,766 parameters. GAP also makes the parameter
  count independent of the image size (the same weights work at 160 or
  224 px) and tolerant to where the leaf sits in the frame.
- **Dropout 0.3 (`model.dropout` in `configs/base.yaml`).** Dropout
  regularises the only fully connected layer, where co-adaptation is most
  likely. It has no parameters and is off at inference.
- **Dense(38) + softmax.** The head outputs one probability per PlantVillage
  class, which is what categorical cross-entropy expects. Macro-F1 is
  computed from the argmax.

## Link to the bf16 study

With He-normal initialisation, a block-4 weight starts at a typical magnitude
of √(2 / 1152) ≈ 0.04. In [2⁻⁵, 2⁻⁴), one bf16 step (ULP) is 2⁻¹² ≈ 2.4e-4, so
updates smaller than half of that, ≈ 1.2e-4, round away. Adam moves each weight
by roughly the learning rate per step:

- **Model A, lr 1e-3:** a step is about 8× half an ULP, so most updates
  survive.
- **Model C, lr 1e-5:** a step is about 1/12 of half an ULP, so most updates
  vanish.

This is why Model A is the "low-freeze" control in H1. When training,
construct it with `variable_groups={"features": [conv/BN layers], "head":
[classifier]}` so that the per-group `freeze_rate_*` metrics show which part
of the network freezes first.

## Training budget (target: under 30 minutes on a Colab T4)

These are estimates. The Experimentation Engineer's
`notebooks/00_colab_timing.ipynb` measures the real numbers.

- **Compute.** A forward pass costs ≈ 0.74G multiply-adds per image
  (43M + 3 × 231M), about 1/6 of ResNet-50 (≈ 4.1G). Training is about 3× the
  forward pass, so ≈ 4.4 GFLOP per image, or ≈ 155 TFLOP per epoch of
  35,000 images. At a realistic 2–3 TFLOPS on a T4 (8.1 TFLOPS fp32 peak),
  that is ≈ 50–80 s per epoch, plus input-pipeline time.
- **Memory.** The activations kept for backprop (conv, BN, ReLU and pool
  outputs, float32) take ≈ 40 MB per image, mostly in block 1, so ≈ 1.3 GB
  for a batch of 32. This fits easily in 16 GB.
- **Plan.** Allow ≈ 2 min per epoch at 224 px, which gives about 15 epochs in
  30 minutes. Early stopping (patience 5) usually ends sooner. At 160 px the
  compute halves ((160/224)² ≈ 0.51), which allows about 25–30 epochs.

## Reference build (to become `build_model_a_scratch` in Phase 3)

```python
inputs = keras.Input(shape=(224, 224, 3), name="image")
x = keras.layers.Rescaling(1.0 / 255, name="rescale")(inputs)
for i, filters in enumerate((32, 64, 128, 256), start=1):
    x = keras.layers.Conv2D(filters, 3, padding="same",
                            kernel_initializer="he_normal", name=f"block{i}_conv")(x)
    x = keras.layers.BatchNormalization(name=f"block{i}_bn")(x)
    x = keras.layers.Activation("relu", name=f"block{i}_relu")(x)
    x = keras.layers.MaxPooling2D(2, name=f"block{i}_pool")(x)
x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
x = keras.layers.Dropout(cfg["model"]["dropout"], name="dropout")(x)
outputs = keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
model = PrecisionModel(inputs, outputs, precision=cfg["precision"]["mode"], ...)
```
