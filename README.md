# yolo-doclaynet

<p align="center">
  🤗 <a href="https://huggingface.co/hantian/yolo-doclaynet">Hugging Face</a>
</p>

<p align="center">predict results by <b>yolov8n-doclaynet</b></p>
<p align="center">
  <img src="./test.png" width="400"  alt="page_0"/>
  <img src="./annotated-test.png" width="400"  alt="page_1"/> 
</p>

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
   of [YOLO models](https://huggingface.co/hantian/yolo-doclaynet): `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`
   and `yolov8x`. (still training)

## How to use?

```python
from ultralytics import YOLO

model = YOLO("{path to model file}")
pred = model("{path to test image}")
print(pred)
```

The predict result refer to this [doc](https://docs.ultralytics.com/modes/predict/#working-with-results)

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
  typically appearing in large font

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

After preparing data, training is super easy. All duty jobs has been done
by [Ultralytics](https://github.com/ultralytics/ultralytics). You can choose base models from
this [link](https://docs.ultralytics.com/models/). I use the YOLOv8 series.

```bash
python train.py {base-model}
```

### eval

TODO

## Result

TODO
