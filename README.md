<div align="center">

# YOLO DocLayNet

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-blue)](https://huggingface.co/hantian/yolo-doclaynet)

</div>

### 🔥 Latest Updates

- **2025/03/10**: Released YOLOv12 models - [**Get YOLOv12x**](https://buymeacoffee.com/ppaanngggg/e/384798)
- **2024/10/07**: Released YOLOv11 models - [**Get YOLOv11x**](https://buymeacoffee.com/ppaanngggg/e/313976)
- **2024/07/10**: Released YOLOv10 models - [**Get YOLOv10x**](https://buymeacoffee.com/ppaanngggg/e/275645)
- **2024/06/21**: Released YOLOv9 models

### 🎯 Model Demo

<div align="center">
  <p float="left">
    <img src="./test.png" width="48%" alt="Original Document"/>
    <img src="./annotated-test.png" width="48%" alt="YOLO Detection Result"/>
  </p>
  <p><em>Document layout detection using <strong>YOLOv8n-DocLayNet</strong></em></p>
</div>

## 📊 Performance Results

### Model Performance Comparison Chart (mAP50-95)

<div align="center">
  <img src="./plot.png" width="800px" alt="Model Performance Comparison Plot"/>
  <p><em>Performance comparison of different YOLO models on DocLayNet test dataset</em></p>
</div>

### Detailed Model Performance Metrics (Parameters/mAP50-95)

| Size/Model | YOLOv12     | YOLOv11     | YOLOv10     | YOLOv9      | YOLOv8      |
| ---------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Nano       | 2.6M/0.756  | 2.6M/0.735  | 2.3M/0.730  | 2.0M/0.737  | 3.2M/0.718  |
| Small      | 9.3M/0.782  | 9.4M/0.767  | 7.2M/0.762  | 7.2M/0.766  | 11.2M/0.752 |
| Medium     | 20.2M/0.788 | 20.1M/0.781 | 15.4M/0.780 | 20.1M/0.775 | 25.9M/0.775 |
| Large      | 26.4M/0.792 | 25.3M/0.793 | 24.4M/0.790 | 25.5M/0.782 | 43.7M/0.783 |
| Extra      | 59.1M/0.794 | 56.9M/0.794 | 29.5M/0.793 | -           | 68.2M/0.787 |

Refer to [Detail Results](#detail-results)

## Why this repo?

RAG (Retrieval Augmented Generation) is widely used today for chatting with documents. But when documents have complex layouts, the performance often suffers. It's hard to properly extract and structure content from these complex documents. This project offers a fast and effective solution to this problem.

1. `YOLO` is a leading object detection model by [Ultralytics](https://github.com/ultralytics/ultralytics). It comes in 5 different sizes and has a robust framework for training and deployment. I picked YOLO because of these strengths.

2. `DocLayNet` is a dataset of 80,863 document pages with human-labeled layout information. It includes many different types of documents and is currently the best dataset available for document layout analysis.

## What I did?

Here's what I did:

1. Created a script that converts DocLayNet data into YOLO's training format
2. Built code for training, testing and running the models
3. Trained and shared YOLO models in all sizes and versions

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

## Detail Results

### YOLOv12 Models

| label          | boxes | yolov12n | yolov12s | yolov12m | yolov12l | yolov12x |
| -------------- | ----- | -------- | -------- | -------- | -------- | -------- |
| Params (M)     |       | 2.6      | 9.3      | 20.2     | 26.4     | 59.1     |
| Caption        | 1542  | 0.744    | 0.763    | 0.776    | 0.78     | 0.774    |
| Footnote       | 387   | 0.671    | 0.712    | 0.717    | 0.711    | 0.696    |
| Formula        | 1966  | 0.688    | 0.72     | 0.734    | 0.742    | 0.763    |
| List-item      | 10521 | 0.828    | 0.845    | 0.851    | 0.85     | 0.841    |
| Page-footer    | 3987  | 0.624    | 0.649    | 0.649    | 0.656    | 0.671    |
| Page-header    | 3365  | 0.737    | 0.774    | 0.772    | 0.794    | 0.793    |
| Picture        | 3497  | 0.765    | 0.799    | 0.793    | 0.798    | 0.812    |
| Section-header | 8544  | 0.732    | 0.751    | 0.76     | 0.764    | 0.766    |
| Table          | 2394  | 0.861    | 0.879    | 0.882    | 0.889    | 0.888    |
| Text           | 29917 | 0.863    | 0.878    | 0.884    | 0.884    | 0.88     |
| Title          | 334   | 0.806    | 0.831    | 0.848    | 0.842    | 0.846    |
| **All**        | 66454 | 0.756    | 0.782    | 0.788    | 0.792    | 0.794    |

### YOLOv11 Models

| label          | boxes | yolov11n | yolov11s | yolov11m | yolov11l | yolov11x |
| -------------- | ----- | -------- | -------- | -------- | -------- | -------- |
| Params (M)     |       | 2.6      | 9.4      | 20.1     | 25.3     | 56.9     |
| Caption        | 1542  | 0.717    | 0.744    | 0.746    | 0.772    | 0.765    |
| Footnote       | 387   | 0.634    | 0.683    | 0.701    | 0.715    | 0.71     |
| Formula        | 1966  | 0.673    | 0.705    | 0.729    | 0.75     | 0.765    |
| List-item      | 10521 | 0.81     | 0.836    | 0.843    | 0.847    | 0.845    |
| Page-footer    | 3987  | 0.591    | 0.621    | 0.653    | 0.678    | 0.684    |
| Page-header    | 3365  | 0.704    | 0.76     | 0.778    | 0.788    | 0.795    |
| Picture        | 3497  | 0.758    | 0.783    | 0.8      | 0.805    | 0.802    |
| Section-header | 8544  | 0.713    | 0.745    | 0.753    | 0.75     | 0.751    |
| Table          | 2394  | 0.846    | 0.874    | 0.88     | 0.891    | 0.89     |
| Text           | 29917 | 0.851    | 0.869    | 0.878    | 0.88     | 0.883    |
| Title          | 334   | 0.793    | 0.817    | 0.832    | 0.844    | 0.848    |
| **All**        | 66454 | 0.735    | 0.767    | 0.781    | 0.793    | 0.794    |

### YOLOv10 Models

| label          | boxes | yolov10n | yolov10s | yolov10m | yolov10b | yolov10l | yolov10x |
| -------------- | ----- | -------- | -------- | -------- | -------- | -------- | -------- |
| Params (M)     |       | 2.3      | 7.2      | 15.4     | 19.1     | 24.4     | 29.5     |
| Caption        | 1542  | 0.713    | 0.738    | 0.761    | 0.762    | 0.772    | 0.77     |
| Footnote       | 387   | 0.642    | 0.681    | 0.713    | 0.72     | 0.722    | 0.725    |
| Formula        | 1966  | 0.648    | 0.698    | 0.727    | 0.715    | 0.736    | 0.76     |
| List-item      | 10521 | 0.803    | 0.833    | 0.845    | 0.844    | 0.851    | 0.849    |
| Page-footer    | 3987  | 0.6      | 0.614    | 0.645    | 0.659    | 0.671    | 0.661    |
| Page-header    | 3365  | 0.699    | 0.761    | 0.765    | 0.774    | 0.779    | 0.79     |
| Picture        | 3497  | 0.749    | 0.778    | 0.79     | 0.803    | 0.8      | 0.806    |
| Section-header | 8544  | 0.71     | 0.729    | 0.742    | 0.744    | 0.743    | 0.748    |
| Table          | 2394  | 0.839    | 0.863    | 0.879    | 0.879    | 0.891    | 0.889    |
| Text           | 29917 | 0.85     | 0.868    | 0.879    | 0.874    | 0.88     | 0.882    |
| Title          | 334   | 0.774    | 0.822    | 0.838    | 0.846    | 0.845    | 0.848    |
| **All**        | 66454 | 0.73     | 0.762    | 0.78     | 0.784    | 0.79     | 0.793    |

### YOLOv9 Models

| label          | boxes | yolov9t | yolov9s | yolov9m | yolov9c |
| -------------- | ----- | ------- | ------- | ------- | ------- |
| Params (M)     |       | 2.0     | 7.2     | 20.1    | 25.5    |
| Caption        | 1542  | 0.68    | 0.735   | 0.749   | 0.746   |
| Footnote       | 387   | 0.638   | 0.684   | 0.693   | 0.689   |
| Formula        | 1966  | 0.678   | 0.719   | 0.737   | 0.752   |
| List-item      | 10521 | 0.802   | 0.827   | 0.838   | 0.843   |
| Page-footer    | 3987  | 0.599   | 0.612   | 0.62    | 0.65    |
| Page-header    | 3365  | 0.731   | 0.77    | 0.77    | 0.785   |
| Picture        | 3497  | 0.764   | 0.789   | 0.787   | 0.796   |
| Section-header | 8544  | 0.72    | 0.736   | 0.742   | 0.741   |
| Table          | 2394  | 0.86    | 0.88    | 0.881   | 0.884   |
| Text           | 29917 | 0.856   | 0.869   | 0.874   | 0.877   |
| Title          | 334   | 0.778   | 0.81    | 0.836   | 0.838   |
| **All**        | 66454 | 0.737   | 0.766   | 0.775   | 0.782   |

### YOLOv8 Models

| label          | boxes | yolov8n | yolov8s | yolov8m | yolov8l | yolov8x |
| -------------- | ----- | ------- | ------- | ------- | ------- | ------- |
| Params (M)     |       | 3.2     | 11.2    | 25.9    | 43.7    | 68.2    |
| Caption        | 1542  | 0.682   | 0.721   | 0.746   | 0.75    | 0.753   |
| Footnote       | 387   | 0.614   | 0.669   | 0.696   | 0.702   | 0.717   |
| Formula        | 1966  | 0.655   | 0.695   | 0.723   | 0.75    | 0.747   |
| List-item      | 10521 | 0.789   | 0.818   | 0.836   | 0.841   | 0.841   |
| Page-footer    | 3987  | 0.588   | 0.61    | 0.64    | 0.641   | 0.655   |
| Page-header    | 3365  | 0.707   | 0.754   | 0.769   | 0.776   | 0.784   |
| Picture        | 3497  | 0.723   | 0.762   | 0.789   | 0.796   | 0.805   |
| Section-header | 8544  | 0.709   | 0.727   | 0.742   | 0.75    | 0.748   |
| Table          | 2394  | 0.82    | 0.854   | 0.88    | 0.885   | 0.886   |
| Text           | 29917 | 0.845   | 0.86    | 0.876   | 0.878   | 0.877   |
| Title          | 334   | 0.762   | 0.806   | 0.83    | 0.846   | 0.84    |
| **All**        | 66454 | 0.718   | 0.752   | 0.775   | 0.783   | 0.787   |
