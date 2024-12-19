import typer
from ultralytics import YOLO, RTDETR


def main(
    base_model: str,
    use_rtdetr: bool = False,
    datasets: str = "./datasets/data.yaml",
    epochs: int = 40,
    imgsz: int = 1024,
    batch: int = 8,
    dropout: float = 0.0,
    seed: int = 0,
    resume: bool = False,
    dfl: float = 1.5,
):
    try:
        from clearml import Task

        Task.init(
            project_name="yolo-doclaynet",
            task_name=f"base-model-{base_model}-epochs-{epochs}-imgsz-{imgsz}-batch-{batch}",
        )
    except ImportError:
        print("clearml not installed")

    if use_rtdetr:
        model = RTDETR(base_model)
    else:
        model = YOLO(base_model)
    results = model.train(
        data=datasets,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        dropout=dropout,
        seed=seed,
        resume=resume,
        dfl=dfl,
    )
    print(results)


if __name__ == "__main__":
    typer.run(main)
