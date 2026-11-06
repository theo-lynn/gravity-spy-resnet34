# Gravity Spy glitch classifier (ResNet-34)

Classifies LIGO glitch spectrograms into 22 classes by fine-tuning an ImageNet-pretrained ResNet-34.
There are two Colab notebooks, which differ only in how training starts:

| Notebook | Training |
| --- | --- |
| [`gravity_spy_resnet34_baseline.ipynb`](gravity_spy_resnet34_baseline.ipynb) | All layers train for 25 epochs |
| [`gravity_spy_resnet34_warmup.ipynb`](gravity_spy_resnet34_warmup.ipynb) | 2 epochs training only `layer4` and the output layer, then all layers for 23 epochs |

## Run it

1. Upload a notebook to [Google Colab](https://colab.research.google.com).
2. Choose Runtime → Change runtime type → T4 GPU.
3. Choose Runtime → Run all.

A run takes about 20 minutes on a T4, including the 3.9 GB download. The data comes from the public Kaggle
copy of the Gravity Spy training set (`tentotheminus9/gravity-spy-gravitational-waves`). No Kaggle credentials
were needed in testing, but if the download asks for them, add `KAGGLE_USERNAME` and `KAGGLE_KEY` to Colab's
Secrets panel.

## Method

- **Data:** 7,967 glitches, each rendered at 4 durations (0.5, 1, 2 and 4 s), using the dataset's own
  split of 5,587 train / 1,200 validation / 1,180 test.
- **Input:** each duration is converted to greyscale and resized to 224×224, then the 4 are stacked as one
  4-channel image.
- **Model:** ResNet-34 with a new 4-channel first layer, initialised from the averaged pretrained RGB
  filters, and a new 22-class output layer.
- **Augmentation:** time shifts of up to ±10% only. Flips and frequency shifts would change what a glitch
  means.
- **Training:** 25 epochs, AdamW with a one-cycle learning rate (peak 1e-3), batch size 64, cross-entropy
  with label smoothing 0.1, and mixed precision on GPU. The weights from the best validation epoch are kept.

## Results

One full run of each notebook on a Colab T4. Test scores use the 1,180 held-out glitches.

| | Baseline | Frozen warm-up |
| --- | --- | --- |
| Test accuracy | **97.80%** | 97.63% |
| Test macro-F1 | **0.9437** | 0.9431 |
| Best validation accuracy | **98.17%** | 98.08% |
| Best epoch (of 25) | 22 | 20 |

**The two are effectively tied.** The gap is 2 test images and 1 validation image, from a single run each,
which is too small to separate from random variation. The frozen warm-up gave no measurable benefit here, and
it didn't remove the validation dips near peak learning rate either: it fell to 68.7% at one point, while the
baseline's lowest point after its first epoch was 88.3%.

### Comparison

| | Baseline | Frozen warm-up |
| --- | --- | --- |
| Correct on test set | 1,154 / 1,180 | 1,152 / 1,180 |
| Classes with the higher F1 | 7 | 3 |
| Classes tied | 12 | 12 |
| Lowest validation accuracy after the first epoch | 88.3% | 68.7% |
| Final train loss | 0.621 | 0.621 |

The baseline's 7 per-class wins are mostly small, and the warm-up's biggest win is on Tomte, a class with only
13 test samples. Since the warm-up adds code without improving results, the baseline is the better default.

### Per-class F1

| Class | Test samples | Baseline | Frozen warm-up |
| --- | --- | --- | --- |
| 1080Lines | 50 | 0.980 | 0.980 |
| 1400Ripples | 9 | 0.947 | 0.947 |
| Air_Compressor | 9 | 0.941 | 0.941 |
| Blip | 273 | 0.989 | 0.991 |
| Chirp | 10 | 1.000 | 1.000 |
| Extremely_Loud | 64 | 0.962 | 0.961 |
| Helix | 42 | 1.000 | 0.988 |
| Koi_Fish | 102 | 0.995 | 0.990 |
| Light_Modulation | 78 | 0.975 | 0.968 |
| Low_Frequency_Burst | 90 | 0.989 | 0.989 |
| Low_Frequency_Lines | 66 | 0.954 | 0.947 |
| No_Glitch | 21 | 0.909 | 0.909 |
| None_of_the_Above | 11 | 0.783 | 0.783 |
| Paired_Doves | 4 | 0.667 | 0.667 |
| Power_Line | 68 | 0.993 | 0.993 |
| Repeating_Blips | 37 | 0.972 | 0.959 |
| Scattered_Light | 67 | 0.985 | 0.970 |
| Scratchy | 50 | 0.971 | 0.980 |
| Tomte | 13 | 0.929 | 0.963 |
| Violin_Mode | 64 | 1.000 | 1.000 |
| Wandering_Line | 7 | 0.833 | 0.833 |
| Whistle | 45 | 0.989 | 0.989 |

The weakest classes are the same in both runs: Paired_Doves, None_of_the_Above and Wandering_Line. They have
very few test samples, so their scores are noisy.

## Trained model

The warm-up model weights (`resnet34_gravityspy_warmup.pt`, 85 MB) are kept locally in `models/`, not in this
repo. Loading them and re-scoring the test set reproduced the 97.63% result. The baseline weights were not kept,
so rerun the baseline notebook to recreate them.
