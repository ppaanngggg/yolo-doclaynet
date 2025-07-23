import typer
from ultralytics import YOLO


def main(
    base_model: str,
    datasets: str = "./datasets/data.yaml",
    epochs: int = 40,
    imgsz: int = 1024,
    batch: int = 8,
    seed: int = 42,
):
    try:
        from clearml import Task

        Task.init(
            project_name="yolo-doclaynet",
            task_name=f"{base_model}-epochs-{epochs}-imgsz-{imgsz}-batch-{batch}",
        )
    except ImportError:
        print("clearml not installed")

    model = YOLO(base_model)
    results = model.train(
        data=datasets,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        seed=seed,
    )
    print(results)


if __name__ == "__main__":
    typer.run(main)
