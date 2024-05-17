import json
import os
from pathlib import Path

import tqdm
import typer
import yaml


def main(root_folder: Path = "./datasets"):
    with open(root_folder / "data.yaml", "w") as f:
        yaml.dump(
            {
                "path": "./",
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {
                    "0": "Caption",
                    "1": "Footnote",
                    "2": "Formula",
                    "3": "List-item",
                    "4": "Page-footer",
                    "5": "Page-header",
                    "6": "Picture",
                    "7": "Section-header",
                    "8": "Table",
                    "9": "Text",
                    "10": "Title",
                },
            },
            f,
        )

    for folder in ["val", "test", "train"]:
        print(f"convert {folder} dataset...")
        os.makedirs(root_folder / "labels" / folder, exist_ok=True)
        os.makedirs(root_folder / "images" / folder, exist_ok=True)
        with open(root_folder / "COCO" / f"{folder}.json") as f:
            bigjson = json.load(f)
        for image in tqdm.tqdm(bigjson["images"], desc="move images..."):
            image_id = image["id"]
            filename = image["file_name"]
            os.rename(
                root_folder / "PNG" / filename,
                root_folder / "images" / folder / f"{image_id}.png",
            )
        for annotation in tqdm.tqdm(bigjson["annotations"], desc="write labels..."):
            image_id = annotation["image_id"]
            filename = f"{image_id}.txt"
            left, top, width, height = annotation["bbox"]
            left /= 1025
            top /= 1025
            width /= 1025
            height /= 1025
            center_x = left + width / 2
            center_y = top + height / 2
            category_id = annotation["category_id"] - 1
            with open(root_folder / "labels" / folder / filename, "a") as f:
                f.write(f"{category_id} {center_x} {center_y} {width} {height}\n")


if __name__ == "__main__":
    typer.run(main)
