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

The data comes from the public Kaggle copy of the Gravity Spy training set
(`tentotheminus9/gravity-spy-gravitational-waves`). If the download asks for credentials, add
`KAGGLE_USERNAME` and `KAGGLE_KEY` to Colab's Secrets panel.

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

## Results so far

**Neither notebook has completed a full training run yet.** The baseline figures below are partial: that
run was on a laptop CPU and stopped after 8 of 25 epochs, so it never reached the low learning rates at the
end of the schedule. They come from its best checkpoint (epoch 6, validation accuracy 95.7%). The warm-up
notebook has only been checked for errors on a tiny subset.

| Test set (1,180 glitches) | Baseline (partial, epoch 6) | Baseline (full) | Warm-up (full) |
| --- | --- | --- | --- |
| Accuracy | 96.2% | Not measured | Not measured |
| Macro-F1 | 0.928 | Not measured | Not measured |

The weakest classes in the partial baseline run are rare or catch-all ones:

| Class | F1 | Test samples |
| --- | --- | --- |
| None_of_the_Above | 0.640 | 11 |
| Paired_Doves | 0.667 | 4 |
| No_Glitch | 0.851 | 21 |
| Wandering_Line | 0.875 | 7 |
