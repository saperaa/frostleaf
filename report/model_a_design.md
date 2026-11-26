# Model A design: custom CNN from scratch (Phase 2, updated for 160 px)

Model A is the "trained from scratch" arm of H1. Its learning rate (1e-3) is
large compared with half a bf16 step of its weights, so it should be far less
affected by bf16 weight freezing than Model C (fine-tuning at 1e-5). It is
built in Phase 3 as `build_model_a_scratch()` in `src/models.py`. This note
fixes the design so that the parameter counts can be checked against
`model.summary()` now.

## Architecture

Input 160 × 160 × 3 (RGB, pixels already in [0, 1] from `src.data.preprocess`), 38 output classes. Four identical conv blocks,
then a small classifier head:

```
image 160x160x3 (already scaled to [0, 1])
  -> [ Conv 3x3, 32  -> BatchNorm -> ReLU -> MaxPool 2x2 ]   block 1  ->  80x80x32
  -> [ Conv 3x3, 64  -> BatchNorm -> ReLU -> MaxPool 2x2 ]   block 2  ->  40x40x64
  -> [ Conv 3x3, 128 -> BatchNorm -> ReLU -> MaxPool 2x2 ]   block 3  ->  20x20x128
  -> [ Conv 3x3, 256 -> BatchNorm -> ReLU -> MaxPool 2x2 ]   block 4  ->  10x10x256
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
- ReLU, MaxPool, GlobalAveragePooling and Dropout have no parameters

| # | Layer | Output shape | Hand count | Params |
|---|---|---|---|---:|
| 0 | Input | 160 × 160 × 3 | – | 0 |
| 1 | block1 Conv 3×3, 32 | 160 × 160 × 32 | 3·3·3·32 + 32 = 864 + 32 | 896 |
| 2 | block1 BatchNorm | 160 × 160 × 32 | 4 · 32 | 128 |
| 3 | block1 ReLU | 160 × 160 × 32 | – | 0 |
| 4 | block1 MaxPool 2×2 | 80 × 80 × 32 | – | 0 |
| 5 | block2 Conv 3×3, 64 | 80 × 80 × 64 | 3·3·32·64 + 64 = 18,432 + 64 | 18,496 |
| 6 | block2 BatchNorm | 80 × 80 × 64 | 4 · 64 | 256 |
| 7 | block2 ReLU | 80 × 80 × 64 | – | 0 |
| 8 | block2 MaxPool 2×2 | 40 × 40 × 64 | – | 0 |
| 9 | block3 Conv 3×3, 128 | 40 × 40 × 128 | 3·3·64·128 + 128 = 73,728 + 128 | 73,856 |
| 10 | block3 BatchNorm | 40 × 40 × 128 | 4 · 128 | 512 |
| 11 | block3 ReLU | 40 × 40 × 128 | – | 0 |
| 12 | block3 MaxPool 2×2 | 20 × 20 × 128 | – | 0 |
| 13 | block4 Conv 3×3, 256 | 20 × 20 × 256 | 3·3·128·256 + 256 = 294,912 + 256 | 295,168 |
| 14 | block4 BatchNorm | 20 × 20 × 256 | 4 · 256 | 1,024 |
| 15 | block4 ReLU | 20 × 20 × 256 | – | 0 |
| 16 | block4 MaxPool 2×2 | 10 × 10 × 256 | – | 0 |
| 17 | GlobalAveragePooling2D | 256 | – | 0 |
| 18 | Dropout 0.3 | 256 | – | 0 |
| 19 | Dense 38, softmax | 38 | 256·38 + 38 = 9,728 + 38 | 9,766 |

Totals:

| | Hand count | `model.summary()` |
|---|---:|---:|
| Conv layers | 896 + 18,496 + 73,856 + 295,168 = 388,416 | |
| BatchNorm layers | 128 + 256 + 512 + 1,024 = 1,920 | |
| Dense head | 9,766 | |
| **Total** | 388,416 + 1,920 + 9,766 = **400,102** | 400,102 |
| **Trainable** | 400,102 − 960 = **399,142** | 399,142 |
| **Non-trainable** (moving mean + variance: 2 × (32+64+128+256)) | **960** | 960 |

Checked on 2026-11-25 with Keras 3.15.1 at 224 px, and again at 160 px without
the Rescaling layer (2026-11-26): every per-layer count and all three totals
match `model.summary()`. No layer's parameters depend on the image size (see
global average pooling below), so the counts are identical at both sizes.

## Why each choice

- **160 × 160 × 3 input.** The Colab timing (#11) moved the project from 224 px
  to 160 px so that Model C fits 8 epochs in a 30-minute run. All three models
  see the same 160 px images and differ only in architecture and training.
- **No rescaling layer.** `src.data.preprocess` already scales pixels to
  [0, 1], and it is the one preprocessing shared by training and `predict.py`.
  An extra Rescaling(1/255) here would divide by 255 twice and feed the network
  near-zero inputs. (An earlier version of this note had one; removed
  2026-11-26.) Nothing is fitted, so there is no leakage either way.
- **3 × 3 kernels, stride 1, `same` padding.** Stacked 3 × 3 convolutions see
  as far as one larger kernel with fewer parameters: two 3 × 3 layers cover
  5 × 5 with 18·C² weights instead of 25·C². `same` padding keeps the size
  halving cleanly at each pool, 160 → 80 → 40 → 20 → 10.
- **Four blocks (the top of the 3–4 range).** Each block grows the receptive
  field. After 3 blocks a feature sees only 22 × 22 px; after 4 it sees
  46 × 46 px, about 29% of the 160 px image's width. That is enough to cover a whole
  lesion, ring or spot pattern. The 4th block costs 295k parameters but adds
  the 256 high-level features the head classifies from. A 5th block would
  shrink the map to 5 × 5 and add about 1.2M parameters, which is more than
  this budget allows.
- **Filters 32 → 64 → 128 → 256 (doubling).** Each pool divides the number of
  positions by 4, and doubling the filters multiplies the conv cost by 4, so
  blocks 2–4 cost the same (≈ 118M multiply-adds each at 160 px). No block is a
  bottleneck. Block 1 starts small (32) because it runs at full 160 × 160
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
- **Global average pooling instead of Flatten.** Flatten of the 10 × 10 × 256
  map gives 25,600 features, and a Dense(38) on it would have
  25,600 · 38 + 38 = 972,838 parameters. That is 2.5× the rest of the model
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

Measured by the Experimentation Engineer in `notebooks/00_colab_timing.ipynb`
(#11): at 224 px Model A took 0.272 s per step, about 306 s per epoch including
validation, and the input pipeline, not the model, set that speed. At 160 px the
compute roughly halves ((160/224)² ≈ 0.51) and the decoded-image cache (#13)
removes most of the input cost after the first epoch, so 30 minutes allows
roughly 6–10 epochs; early stopping (patience 5) may end sooner.

## Reference build (to become `build_model_a_scratch` in Phase 3)

```python
size = cfg["data"]["image_size"]                    # 160
inputs = keras.Input(shape=(size, size, 3), name="image")  # already in [0, 1]
x = inputs
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
