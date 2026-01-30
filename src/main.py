import os
import tempfile
import cv2
import pandas as pd
import supervisely_lib as sly

from quality import analyze_image

my_app = sly.AppService()
api = sly.Api.from_env()

TASK_ID = int(os.environ["TASK_ID"])
TEAM_ID = int(os.environ["context.teamId"])

# ---- Project / Dataset resolution (robust) ----
PROJECT_ID = (
    os.environ.get("context.projectId")
    or os.environ.get("context.slyProjectId")
    or os.environ.get("state.slyProjectId")
)

if PROJECT_ID is None:
    raise RuntimeError(
        "Project ID not found. App must be launched from Project/Dataset context menu."
    )

PROJECT_ID = int(PROJECT_ID)
DATASET_ID = os.environ.get("context.datasetId")

# ---- Modal values ----
BLUR_TH = float(os.environ.get("state.blurTh", 100))
LOW_BRIGHTNESS = float(os.environ.get("state.lowBrightness", 60))
HIGH_BRIGHTNESS = float(os.environ.get("state.highBrightness", 200))
GRAYSCALE_TOL = float(os.environ.get("state.grayscaleTol", 2))
EXPORT_CSV = os.environ.get("state.exportCsv", "true") == "true"


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
            tmp_path = os.path.join(
                tempfile.gettempdir(),
                f"{img_info.id}_{img_info.name}"
            )

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
        api.file.upload(TEAM_ID, csv_path, remote_path)

    sly.logger.info("QC finished", extra={"images": len(rows)})
    my_app.stop()


def main():
    my_app.run(initial_events=[{"command": "do"}])


if __name__ == "__main__":
    sly.main_wrapper("main", main)
