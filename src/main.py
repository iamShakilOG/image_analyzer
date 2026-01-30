import os
import tempfile
import cv2
import pandas as pd
import supervisely_lib as sly

from quality import analyze_image


# -----------------------------------
# APP SERVICE (MODAL APP)
# -----------------------------------
my_app = sly.AppService()
api = sly.Api.from_env()

TASK_ID = int(os.environ["TASK_ID"])
TEAM_ID = int(os.environ["context.teamId"])
PROJECT_ID = int(os.environ["context.projectId"])
DATASET_ID = os.environ.get("context.datasetId")


# -----------------------------------
# READ MODAL STATE
# -----------------------------------
BLUR_TH = float(os.environ.get("state.blurTh", 100))
LOW_BRIGHTNESS = float(os.environ.get("state.lowBrightness", 60))
HIGH_BRIGHTNESS = float(os.environ.get("state.highBrightness", 200))
GRAYSCALE_TOL = float(os.environ.get("state.grayscaleTol", 2))
EXPORT_CSV = os.environ.get("state.exportCsv", "true") == "true"


@my_app.callback("do")
@sly.timeit
def do(**kwargs):
    sly.logger.info(
        "Starting Image Quality Analysis",
        extra={
            "project_id": PROJECT_ID,
            "dataset_id": DATASET_ID,
        },
    )

    cfg = {
        "blur_th": BLUR_TH,
        "low_brightness": LOW_BRIGHTNESS,
        "high_brightness": HIGH_BRIGHTNESS,
        "grayscale_tol": GRAYSCALE_TOL,
    }

    # Decide datasets
    if DATASET_ID not in (None, "", "None"):
        dataset_ids = [int(DATASET_ID)]
    else:
        dataset_ids = [d.id for d in api.dataset.get_list(PROJECT_ID)]

    rows = []
    stats = {"total": 0, "blur": 0, "dark": 0, "bright": 0, "gray": 0}

    for ds_id in dataset_ids:
        for img_info in api.image.get_list(ds_id):
            tmp = os.path.join(tempfile.gettempdir(), img_info.name)
            api.image.download(img_info.id, tmp)

            img = cv2.imread(tmp)
            if img is None:
                continue

            res = analyze_image(img, cfg)

            stats["total"] += 1
            stats["blur"] += int(res["blur"])
            stats["dark"] += int(res["low_brightness"])
            stats["bright"] += int(res["high_brightness"])
            stats["gray"] += int(res["grayscale"])

            rows.append({"image": img_info.name, **res})

    sly.logger.info("Image Quality Summary", extra=stats)

    # -----------------------------------
    # EXPORT CSV
    # -----------------------------------
    if EXPORT_CSV and rows:
        df = pd.DataFrame(rows)
        local_csv = "/sly-app-data/image_quality_report.csv"
        df.to_csv(local_csv, index=False)

        remote_path = f"/image_quality_analyzer/{PROJECT_ID}_report.csv"
        api.file.upload(TEAM_ID, local_csv, remote_path)

        sly.logger.info("CSV uploaded", extra={"path": remote_path})

    # -----------------------------------
    # SET OUTPUT & FINISH
    # -----------------------------------
    api.task.set_output_text(
        TASK_ID,
        f"""
        Images analyzed: {stats['total']}
        Blurred: {stats['blur']}
        Too dark: {stats['dark']}
        Too bright: {stats['bright']}
        Grayscale: {stats['gray']}
        """,
    )

    my_app.stop()


def main():
    my_app.run(initial_events=[{"command": "do"}])


if __name__ == "__main__":
    sly.main_wrapper("main", main)
