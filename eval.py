import typer
from ultralytics import YOLO


def main(
    model: str,
    datasets: str = "./datasets/data.yaml",
    split: str = "test",
    batch: int = 8,
):
    model = YOLO(model)
    metrics = model.val(data=datasets, split=split, batch=batch)
    print(metrics)


if __name__ == "__main__":
    typer.run(main)
