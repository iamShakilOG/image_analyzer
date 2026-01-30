import os
import tempfile
import cv2
import pandas as pd
import supervisely_lib as sly

from quality import analyze_image

my_app = sly.AppService()
api = sly.Api.from_env()

TASK_ID = int(os.environ["TASK_ID"])
PROJECT_ID = int(os.environ["modal.state.slyProjectId"])
DATASET_ID = os.environ.get("context.datasetId")

# ---- values from modal.html ----
BLUR_TH = int(os.environ.get("modal.state.blurTh", 100))
LOW_BRIGHTNESS = int(os.environ.get("modal.state.lowBrightness", 60))
HIGH_BRIGHTNESS = int(os.environ.get("modal.state.highBrightness", 200))
GRAYSCALE_TOL = int(os.environ.get("modal.state.grayscaleTol", 2))
EXPORT_CSV = os.environ.get("modal.state.exportCsv", "true") == "true"


@my_app.callback("do")
@sly.timeit
def do(**kwargs):
    dataset_id = int(DATASET_ID) if DATASET_ID not in (None, "", "None") else None

    cfg = {
        "blur_th": BLUR_TH,
        "low_brightness": LOW_BRIGHTNESS,
        "high_brightness": HIGH_BRIGHTNESS,
        "grayscale_tol": GRAYSCALE_TOL,
    }

    rows = []
    datasets = (
        [dataset_id]
        if dataset_id
        else [d.id for d in api.dataset.get_list(PROJECT_ID)]
    )

    for ds_id in datasets:
        for img_info in api.image.get_list(ds_id):
            tmp_path = os.path.join(tempfile.gettempdir(), img_info.name)
            api.image.download(img_info.id, tmp_path)

            img = cv2.imread(tmp_path)
            if img is None:
                continue

            res = analyze_image(img, cfg)
            rows.append({"image": img_info.name, **res})

    if EXPORT_CSV and rows:
        df = pd.DataFrame(rows)
        csv_path = "/tmp/image_quality_report.csv"
        df.to_csv(csv_path, index=False)

        remote_path = f"/image_quality_analyzer/{PROJECT_ID}_report.csv"
        api.file.upload(TASK_ID, csv_path, remote_path)

    sly.logger.info("QC finished", extra={"images": len(rows)})
    my_app.stop()


def main():
    my_app.run(initial_events=[{"command": "do"}])


if __name__ == "__main__":
    sly.main_wrapper("main", main)
