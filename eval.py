import typer
from ultralytics import YOLO


def main(model: str, datasets: str = "./datasets/data.yaml", split: str = "test"):
    model = YOLO(model)
    metrics = model.val(data=datasets, split=split)
    print(metrics)


if __name__ == "__main__":
    typer.run(main)
