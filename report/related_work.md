# Related work and novelty (Phase 2)

Checked 2026-11-26. Every entry below was looked up at its source; quote nothing
from these papers in the report without re-reading the original.

## Low-precision training (what is already known)

| Paper | What it shows | How we use it |
|---|---|---|
| Gupta, Agrawal, Gopalakrishnan, Narayanan. *Deep Learning with Limited Numerical Precision.* ICML 2015, PMLR 37:1737–1746. [arXiv:1502.02551](https://arxiv.org/abs/1502.02551) | With 16-bit fixed-point numbers, the rounding scheme decides whether training works; stochastic rounding makes it work with little or no accuracy loss. | Origin of stochastic rounding for training (our `bf16_sr` mode). |
| Micikevicius et al. *Mixed Precision Training.* ICLR 2018. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740) | Keeps an FP32 master copy of the weights because small updates (weight much larger than update) are lost in FP16. | The standard fix we deliberately remove; our "mixed" control condition (Keras `mixed_bfloat16`). |
| Kalamkar et al. *A Study of BFLOAT16 for Deep Learning Training.* 2019. [arXiv:1905.12322](https://arxiv.org/abs/1905.12322) | BF16 mixed-precision training matches FP32 across many domains without hyperparameter changes; discusses FP32→BF16 rounding modes. | Why BF16 is worth studying; their setting keeps FP32 master weights. |
| Zamirai, Zhang, Aberger, De Sa. *Revisiting BFloat16 Training.* 2020 (v2 2021). [arXiv:2010.06192](https://arxiv.org/abs/2010.06192) | **Closest prior work.** In pure 16-bit (BF16) training, nearest rounding of weight updates cancels small updates and hurts accuracy; stochastic rounding or Kahan summation on the weight update recovers it, matching 32-bit training across seven applications. | Establishes the effect and both fixes. We must cite it as the source of the phenomenon. |

## Plant disease datasets and the lab-to-field gap

| Paper | Use |
|---|---|
| Hughes & Salathé, 2015, [arXiv:1511.08060](https://arxiv.org/abs/1511.08060); Mohanty, Hughes & Salathé, *Frontiers in Plant Science* 2016 | PlantVillage dataset (cite both, as the dataset card asks). |
| Singh et al. *PlantDoc: A Dataset for Visual Plant Disease Detection.* 2019, [arXiv:1911.10317](https://arxiv.org/abs/1911.10317) | External field-photo test set. |
| Ahmad, El Gamal, Saraswat. *Toward Generalization of Deep Learning-Based Plant Disease Identification Under Controlled and Field Conditions.* IEEE Access 2023, [doi:10.1109/ACCESS.2023.3240100](https://doi.org/10.1109/ACCESS.2023.3240100) | Lab-trained models generalise worse to field images; motivates our PlantDoc evaluation. |
| Benabbas et al. *Rethinking Plant Disease Diagnosis: Bridging the Academic–Practical Gap with Vision Transformers and Zero-Shot Learning.* 2025, [arXiv:2511.18989](https://arxiv.org/abs/2511.18989) | Recent statement of the PlantVillage-to-field-photo gap. |

## What is new in FrostLeaf (state it exactly this way)

We do **not** claim to discover that BF16 nearest rounding loses small weight
updates, or that stochastic rounding fixes it: Zamirai et al. (2020) showed both.
Our contribution is narrower and measurable:

1. **Where on the transfer-learning ladder the loss happens.** The same task
   trained from scratch (Model A, large LR), with a frozen backbone (Model B) and
   fine-tuned (Model C, small LR), with the share of each weight update that
   survives (`update_ratio`) measured per step and per parameter group
   (backbone vs head).
2. **The setting is isolated:** BF16 weight storage with FP32 optimizer state, so
   the effect is weight-update underflow alone, not rounding of the optimizer
   moments.
3. **Optimizer and weight-decay interaction:** SGD vs momentum vs Adam (H3) and
   decay applied as a separate rounded step vs fused with the update (H4).
4. **A field-photo test:** whether the freeze matters on PlantDoc, where the gap
   between Models B and C is larger than on PlantVillage.
