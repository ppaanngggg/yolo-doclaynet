import typer
from ultralytics import YOLO


def main(
    base_model: str = "yolov8n.pt",
    datasets: str = "../datasets/data.yaml",
    epochs: int = 100,
    imgsz: int = 1024,
    batch: int = 16,
):
    try:
        from clearml import Task

        Task.init(project_name="yolo-doclaynet", task_name="yolov8n")
    except ImportError:
        print("clearml not installed")

    model = YOLO(base_model)
    results = model.train(data=datasets, epochs=epochs, imgsz=imgsz, batch=batch)
    print(results)


if __name__ == "__main__":
    typer.run(main)
