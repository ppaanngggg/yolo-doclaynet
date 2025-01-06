# yolo-doclaynet

<p align="left">
<a href="https://huggingface.co/hantian/yolo-doclaynet">🤗 Hugging Face</a> | 
<a href="https://buymeacoffee.com/ppaanngggg/e/313968">📁 YOLOv11l</a> |
<a href="https://buymeacoffee.com/ppaanngggg/e/313976">📁 YOLOv11x</a> |
<a href="https://buymeacoffee.com/ppaanngggg/e/275642">📁 YOLOv10l</a> |
<a href="https://buymeacoffee.com/ppaanngggg/e/275645">📁 YOLOv10x</a> |
<a href="https://buymeacoffee.com/ppaanngggg/e/268779">📁 YOLOv9c</a> |
<a href="https://buymeacoffee.com/ppaanngggg/e/257457">📁 YOLOv8l</a> |
<a href="https://buymeacoffee.com/ppaanngggg/e/257777">📁 YOLOv8x</a>
</p>

**👏 Update 2024/10/07 - Add YOLOv11 models.**

**👏 Update 2024/07/10 - Add YOLOv10 models.**

**👏 Update 2024/06/21 - Add YOLOv9 models.**

<p align="left">
  <img src="./test.png" width="400"  alt="page_0"/>
  <img src="./annotated-test.png" width="400"  alt="page_1"/> 
</p>
<p align="left">predict results by <b>yolov8n-doclaynet</b></p>

## Why this repo?

You know that RAG is very popular these days. There are many applications that support talking to documents. However,
there is a huge performance drop when talking to a complex document due to the complex structures. So it's a challenge
to extract content from complex document and organize it into parsable form. This repo aims to solve this challenge with
a fast and good performance method.

1. `YOLO` is the most advenced detect model developed by [Ultralytics](https://github.com/ultralytics/ultralytics). YOLO
   has 5 different sizes of base model and a super powerful framework for training and deployment. So I chose YOLO to
   solve this challenge.
2. `DocLayNet` is a human-annotated document layout segmentation dataset containing 80863 pages from a broad variety of
   document sources. As far as I know, it's the most qualified document layout analysis dataset.

## What I did?

1. Offer a script to turn DocLayNet dataset into YOLO detect training ready dataset.
2. Offer train, eval and serve codes.
3. Train and release 5 different sizes
   of YOLOv8 models: `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`
   and `yolov8x`.
    - `yolov8n`, `yolov8s` and `yolov8m` can be found on [HuggingFace](https://huggingface.co/hantian/yolo-doclaynet).
    - `yolov8l` and `yolov8x` are only slightly better than `yolov8m`. If you really want to try, please buy
      from [yolov8l](https://buymeacoffee.com/ppaanngggg/e/257457)
      and [yolov8x](https://buymeacoffee.com/ppaanngggg/e/257777), as I rent GPUs to train them.

## How to use?

```python
from ultralytics import YOLO

model = YOLO("{path to model file}")
pred = model("{path to test image}")
print(pred)
```

The definition of predict result please refer to
the [doc](https://docs.ultralytics.com/modes/predict/#working-with-results).

### Server

You can simply `python main.py` to serve the model. Open http://localhost:8000/redoc check the API.

## Dataset

DocLayNet can be found more details and download at this [link](https://github.com/DS4SD/DocLayNet). It has 11 labels:

- **Text**: Regular paragraphs.
- **Picture**: A graphic or photograph.
- **Caption**: Special text outside a picture or table that introduces this picture or
  table.
- **Section-header**: Any kind of heading in the text, except overall document title.
- **Footnote**: Typically small text at the bottom of a page, with a number or symbol
  that is referred to in the text above.
- **Formula**: Mathematical equation on its own line.
- **Table**: Material arranged in a grid alignment with rows and columns, often
  with separator lines.
- **List-item**: One element of a list, in a hanging shape, i.e., from the second line
  onwards the paragraph is indented more than the first line.
- **Page-header**: Repeating elements like page number at the top, outside of the
  normal text flow.
- **Page-footer**: Repeating elements like page number at the bottom, outside of the
  normal text flow.
- **Title**: Overall title of a document, (almost) exclusively on the first page and
  typically appearing in large font.

### Prepare data

1. download DocLayNet dataset by
   this [link](https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip)
2. unzip to `datasets` folder
3. use my convert script to make datasets ready for training

```bash
wget https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip
mkdir datasets
mv DocLayNet_core.zip datasets/
cd datasets/ && unzip DocLayNet_core.zip && rm DocLayNet_core.zip
cd ../
python convert_dataset.py
```

## Train & Eval

### train

After preparing data, thanks to [Ultralytics](https://github.com/ultralytics/ultralytics), training is super easy. You
can choose base models from this [link](https://docs.ultralytics.com/models/). I use the YOLOv8 series.

```bash
python train.py {base-model}
```

### Eval

After training, you can evaluate your best model on test split.

```bash
python eval.py {path-to-your-model}
```

## Result

* Figure of overall `mAP50-95` on `test` between different models.

<img src="./plot.png" width="800"  alt="plot"/>

* Full table of `mAP50-95` on `test` compare between different models.

| label          | boxes | yolov8n | yolov9t | yolov10n | yolov11n | yolov8s | yolov9s | yolov10s | yolov11s | yolov8m | yolov9m | yolov10m | yolov11m | yolov10b | yolov8l | yolov9c | yolov10l | yolov11l | yolov8x | yolov10x | yolov11x |
|----------------|-------|---------|---------|----------|----------|---------|---------|----------|----------|---------|---------|----------|----------|----------|---------|---------|----------|----------|---------|----------|----------|
| Params (M)     |       | 3.2     | 2.0     | 2.3      | 2.6      | 11.2    | 7.2     | 7.2      | 9.4      | 25.9    | 20.1    | 15.4     | 20.1     | 19.1     | 43.7    | 25.5    | 24.4     | 25.3     | 68.2    | 29.5     | 56.9     |
| Caption        | 1542  | 0.682   | 0.68    | 0.713    | 0.717    | 0.721   | 0.735   | 0.738    | 0.744    | 0.746   | 0.749   | 0.761    | 0.746    | 0.762    | 0.75    | 0.746   | 0.772    | 0.772    | 0.753   | 0.77     | 0.765    |
| Footnote       | 387   | 0.614   | 0.638   | 0.642    | 0.634    | 0.669   | 0.684   | 0.681    | 0.683    | 0.696   | 0.693   | 0.713    | 0.701    | 0.72     | 0.702   | 0.689   | 0.722    | 0.715    | 0.717   | 0.725    | 0.71     |
| Formula        | 1966  | 0.655   | 0.678   | 0.648    | 0.673    | 0.695   | 0.719   | 0.698    | 0.705    | 0.723   | 0.737   | 0.727    | 0.729    | 0.715    | 0.75    | 0.752   | 0.736    | 0.75     | 0.747   | 0.76     | 0.765    |
| List-item      | 10521 | 0.789   | 0.802   | 0.803    | 0.81     | 0.818   | 0.827   | 0.833    | 0.836    | 0.836   | 0.838   | 0.845    | 0.843    | 0.844    | 0.841   | 0.843   | 0.851    | 0.847    | 0.841   | 0.849    | 0.845    |
| Page-footer    | 3987  | 0.588   | 0.599   | 0.6      | 0.591    | 0.61    | 0.612   | 0.614    | 0.621    | 0.64    | 0.62    | 0.645    | 0.653    | 0.659    | 0.641   | 0.65    | 0.671    | 0.678    | 0.655   | 0.661    | 0.684    |
| Page-header    | 3365  | 0.707   | 0.731   | 0.699    | 0.704    | 0.754   | 0.77    | 0.761    | 0.76     | 0.769   | 0.77    | 0.765    | 0.778    | 0.774    | 0.776   | 0.785   | 0.779    | 0.788    | 0.784   | 0.79     | 0.795    |
| Picture        | 3497  | 0.723   | 0.764   | 0.749    | 0.758    | 0.762   | 0.789   | 0.778    | 0.783    | 0.789   | 0.787   | 0.79     | 0.8      | 0.803    | 0.796   | 0.796   | 0.8      | 0.805    | 0.805   | 0.806    | 0.802    |
| Section-header | 8544  | 0.709   | 0.72    | 0.71     | 0.713    | 0.727   | 0.736   | 0.729    | 0.745    | 0.742   | 0.742   | 0.742    | 0.753    | 0.744    | 0.75    | 0.741   | 0.743    | 0.75     | 0.748   | 0.748    | 0.751    |
| Table          | 2394  | 0.82    | 0.86    | 0.839    | 0.846    | 0.854   | 0.88    | 0.863    | 0.874    | 0.88    | 0.881   | 0.879    | 0.88     | 0.879    | 0.885   | 0.884   | 0.891    | 0.891    | 0.886   | 0.889    | 0.89     |
| Text           | 29917 | 0.845   | 0.856   | 0.85     | 0.851    | 0.86    | 0.869   | 0.868    | 0.869    | 0.876   | 0.874   | 0.879    | 0.878    | 0.874    | 0.878   | 0.877   | 0.88     | 0.88     | 0.877   | 0.882    | 0.883    |
| Title          | 334   | 0.762   | 0.778   | 0.774    | 0.793    | 0.806   | 0.81    | 0.822    | 0.817    | 0.83    | 0.836   | 0.838    | 0.832    | 0.846    | 0.846   | 0.838   | 0.845    | 0.844    | 0.84    | 0.848    | 0.848    |
| **All**        | 66454 | 0.718   | 0.737   | 0.73     | 0.735    | 0.752   | 0.766   | 0.762    | 0.767    | 0.775   | 0.775   | 0.78     | 0.781    | 0.784    | 0.783   | 0.782   | 0.79     | 0.793    | 0.787   | 0.793    | 0.794    |
