import os
import supervisely_lib as sly
import pandas as pd
import cv2

from ui import (
    blur_th,
    low_brightness,
    high_brightness,
    grayscale_tol,
    export_csv,
)
from quality import analyze_image


my_app = sly.AppService()
api = sly.Api.from_env()

TASK_ID = int(os.environ["TASK_ID"])
PROJECT_ID = int(os.environ["context.projectId"])
DATASET_ID = os.environ.get("context.datasetId")  # may be missing

@my_app.callback("run")
@sly.timeit
def run(**kwargs):
    dataset_id = int(DATASET_ID) if DATASET_ID not in (None, "", "None") else None

    cfg = {
        "blur_th": blur_th.get_value(),
        "low_brightness": low_brightness.get_value(),
        "high_brightness": high_brightness.get_value(),
        "grayscale_tol": grayscale_tol.get_value(),
    }

    rows = []

    datasets = [dataset_id] if dataset_id else [d.id for d in api.dataset.get_list(PROJECT_ID)]

    for ds_id in datasets:
        images = api.image.get_list(ds_id)

        for img_info in images:
            local_path = api.image.download_path(img_info.id)
            img = cv2.imread(local_path)

            res = analyze_image(img, cfg)

            # OPTIONAL: add tags to image (only if tags exist in project meta)
            # If you want tags, we can add TagMeta automatically.
            rows.append({"image": img_info.name, **res})

    if export_csv.is_checked():
        df = pd.DataFrame(rows)
        csv_path = "/tmp/image_quality_report.csv"
        df.to_csv(csv_path, index=False)

        # upload csv to Team Files (easiest)
        remote_path = f"/image_quality_analyzer/{PROJECT_ID}_report.csv"
        api.file.upload(TASK_ID, csv_path, remote_path)
        sly.logger.info("CSV uploaded", extra={"remote_path": remote_path})

    sly.logger.info("Done", extra={"count": len(rows)})
    my_app.stop()


def main():
    my_app.run(initial_events=[{"command": "run"}])


if __name__ == "__main__":
    sly.main_wrapper("main", main)
