import os.path

import cv2
import typer
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, Colors


def main(model: str, image: str, line_width: int = 2, font_size: int = 8):
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    model = YOLO(model)
    result = model.predict(img)[0]
    height = result.orig_shape[0]
    width = result.orig_shape[1]

    colors = Colors()
    annotator = Annotator(img, line_width=line_width, font_size=font_size)
    for label, box in zip(result.boxes.cls.tolist(), result.boxes.xyxyn.tolist()):
        label = int(label)
        annotator.box_label(
            [box[0] * width, box[1] * height, box[2] * width, box[3] * height],
            result.names[label],
            color=colors(label, bgr=True),
        )
    annotator.save(
        os.path.join(os.path.dirname(image), "annotated-" + os.path.basename(image))
    )


if __name__ == "__main__":
    typer.run(main)
