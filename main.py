from threading import Semaphore

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from ultralytics import YOLO


class Conf(BaseSettings):
    model_path: str = Field(default="yolo-doclaynet.pt", description="Model path")
    max_connections: int = Field(
        default=10, description="Maximum number of connections"
    )
    port: int = Field(default=8000, description="Port number")


conf = Conf()
app = FastAPI()
model = YOLO(conf.model_path)
semaphore = Semaphore(conf.max_connections)


class LabelBox(BaseModel):
    label: str = Field(example="Text", description="Label of the object")
    box: list[float] = Field(
        example=[0.0, 0.0, 0.0, 0.0], description="Bounding box coordinates"
    )


class DetectResponse(BaseModel):
    label_boxes: list[LabelBox] = Field(
        example=[{"label": "Text", "box": [0.0, 0.0, 0.0, 0.0]}],
        description="List of detected objects",
    )
    speed: dict = Field(
        example={"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0},
        description="Speed in milliseconds",
    )


@app.post("/api/detect")
def detect(image: UploadFile) -> DetectResponse:
    logger.info(f"Received image: {image.filename}, {image.size}")
    with semaphore:
        image = cv2.imdecode(
            np.frombuffer(image.file.read(), np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        result = model.predict(image, verbose=False)[0]
    height = result.orig_shape[0]
    width = result.orig_shape[1]
    label_boxes = []
    for label, box in zip(result.boxes.cls.tolist(), result.boxes.xyxyn.tolist()):
        label_boxes.append(
            LabelBox(
                label=result.names[int(label)],
                box=[box[0] * width, box[1] * height, box[2] * width, box[3] * height],
            )
        )
    logger.info(
        f"Detected objects: {len(label_boxes)}, Image size: {width}x{height}, Speed: {result.speed}"
    )
    return DetectResponse(label_boxes=label_boxes, speed=result.speed)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=conf.port)
