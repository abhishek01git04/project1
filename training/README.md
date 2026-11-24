# Training our own models

Four models: two built **from scratch** on a laptop CPU, and two **fine-tuned** from
pretrained networks on a free Colab GPU. Every script saves the same metrics (accuracy,
precision, recall, F1, ROC-AUC, confusion matrix) to `models/<name>/`, so the models can
be compared directly.

| # | Model | Task | Approach | Where | Time |
|---|---|---|---|---|---|
| 1 | `text-baseline` | Fake news | TF-IDF (1–2-grams) + Logistic Regression | Laptop CPU | ~5 min |
| 2 | `image-cnn` | Deepfake faces | **TruthNet**: our own 5-block CNN, ~1M parameters | Laptop CPU | ~20–40 min |
| 3 | `text-distilbert` | Fake news | DistilBERT, fine-tuned | Colab GPU | ~25 min |
| 4 | `image-efficientnet` | Deepfake faces | EfficientNet-B0, fine-tuned | Colab GPU | ~30 min |

## 1. Get the datasets (free Kaggle account)

| Dataset | Kaggle page | Put it at |
|---|---|---|
| **WELFake**: 72k news articles labelled real/fake | kaggle.com/datasets/saurabhshahane/fake-news-classification | `data/WELFake_Dataset.csv` |
| **140k Real and Fake Faces**: 70k real (Flickr) + 70k StyleGAN faces | kaggle.com/datasets/xhlulu/140k-real-and-fake-faces | `data/real_vs_fake/real-vs-fake/{train,valid,test}/{fake,real}` |

Download each one with the **Download** button and unzip it into `project1/data/`.

**Our setup** uses a smaller face dataset, *StyleGAN and StyleGAN2 combined* (7,000 fake and
5,890 real, with no predefined split). It was split with:

```powershell
ai-gateway\.venv\Scripts\python training\split_images.py --src "data\faces-raw\Final Dataset" --dst data\faces
```

This dataset contains 5 augmented copies of some photos (`*_aug_0..4.jpg`). The split keeps
all copies of a photo in the same split; otherwise the test set would contain near-duplicates
of training images and the reported accuracy would be inflated.

## 2. Train models 1 and 2 on this PC

Run from the project root (`C:\Users\BIT\project1`). The libraries are already installed in
the gateway's virtual environment. For a fresh setup, run `ai-gateway\.venv\Scripts\python -m pip install -r training\requirements.txt` first.

```powershell
ai-gateway\.venv\Scripts\python training\text_baseline.py --data data\WELFake_Dataset.csv
ai-gateway\.venv\Scripts\python training\image_cnn.py --data data\real_vs_fake\real-vs-fake
```

`image_cnn.py` uses 20,000 training images and 6 epochs by default. Use `--train-limit 0`
(all 100k images) and `--epochs 10` for better accuracy if you can leave it running.

## 3. Train models 3 and 4 on Google Colab (free GPU)

1. `ai-gateway\.venv\Scripts\python training\make_colab_bundle.py` creates `training\colab_bundle.zip`.
2. Open colab.research.google.com, choose **File → Upload notebook**, and pick `training/colab_finetune.ipynb`.
3. **Runtime → Change runtime type → T4 GPU**, then run all the cells. When asked, upload
   `colab_bundle.zip` and your `kaggle.json`.
4. Unzip the downloaded `trained_models.zip` into `project1/`.

## 4. Use the models in the gateway

Restart the gateway. It automatically uses the best model available in `models/`:
fine-tuned if present, otherwise from scratch. To choose one yourself, set `TEXT_MODEL` and
`IMAGE_MODEL` in `ai-gateway/.env` (see `.env.example`). `GET /health` shows which models are active.

## 5. Compare the results (for the report)

```powershell
ai-gateway\.venv\Scripts\python training\compare.py
```
This prints a Markdown table of all trained models. Each `models/<name>/` folder also contains
`confusion_matrix.png`, `metrics.json` and, for the neural networks, `training_curve.png`.
`text-baseline/top_terms.json` lists the words the model treats as the strongest evidence of fake or real news.

## 6. Image forensics: benchmark and calibrate Smaug, TruFor and UnivFD

Uploaded images are scored by three models that measure different things, so each one is benchmarked on
its own and the scores are combined through a fitted calibration, never a plain average.

1. **UnivFD:** `models/image-univfd/fc_weights.pth` comes from
   [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) (MIT). CLIP ViT-L/14
   downloads on first use (1.7 GB).
2. **TruFor:** unzip `trufor.pth.tar` from
   [TruFor_weights.zip](https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip) into
   `models/image-trufor/`. **Nonprofit use only**; see `ai-gateway/app/vendor/trufor/README.md`.
3. **Test images:** the `test` split of `julienlucas/midjourney-dalle-sd-nanobananapro-dataset` (whole AI
   images vs real) and [CocoGlide](https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip) (the same
   photos real and with one region inpainted by AI).

Run one model per process, then fit. Each benchmark can be stopped and resumed:

```powershell
ai-gateway\.venv\Scripts\python training\benchmark_image_models.py --model univfd --parquet <folder> --cocoglide <folder>
ai-gateway\.venv\Scripts\python training\benchmark_image_models.py --model trufor --parquet <folder> --cocoglide <folder>
ai-gateway\.venv\Scripts\python training\benchmark_image_models.py --model smaug --parquet <folder> --cocoglide <folder>
ai-gateway\.venv\Scripts\python training\fit_image_fusion.py
```

`models/image-fusion/benchmark.json` holds the results on held-out images. `calibration.json` is written only
if the fusion beats Smaug alone, and the gateway picks it up without a restart. Until that file exists, Smaug
alone decides the score, and UnivFD and TruFor are shown in the report but not counted.

## Methodology notes (worth including in the report)

- **Data leakage fix:** most real WELFake articles are Reuters copy starting with
  "CITY (Reuters) -". `ai-gateway/app/textclean.py` strips agency names, datelines and URLs
  before training *and* at inference. Otherwise a model can score ~99% just by spotting the
  word "Reuters". Duplicate articles are also removed before splitting.
- **Label auto-detection:** sources disagree on whether `label=0` or `label=1` means fake in
  WELFake, so the scripts infer it from the Reuters datelines and print what they chose.
  Override it with `--fake-label`.
- **Fair comparison:** both text models use the same fixed 80/10/10 split (seed 42).
  `--train-limit` shrinks only the training set, never the test set.
- **Same preprocessing in training and serving:** TruthNet uses `app/cnn.py:to_tensor` in
  both places, and the fine-tuned models use their saved Hugging Face processor.
- **Limitations:** the image models learn *StyleGAN face* artefacts. They are not general
  detectors for diffusion images (Midjourney, DALL·E) or for face swaps in video. Text models
  trained on 2016–2018 political news may not transfer well to other topics or languages.
