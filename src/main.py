import os
import tempfile
import cv2
import pandas as pd
import supervisely_lib as sly

from quality import analyze_image

# ---- App init ----
my_app = sly.AppService()
api = sly.Api.from_env()

TASK_ID = int(os.environ["TASK_ID"])
TEAM_ID = sly.env.team_id()

# ✅ Correct context resolution (LEGACY SAFE)
PROJECT_ID = sly.env.project_id()
DATASET_ID = sly.env.dataset_id(raise_not_found=False)

# ---- Modal values ----
BLUR_TH = float(os.environ.get("state.blurTh", 100))
LOW_BRIGHTNESS = float(os.environ.get("state.lowBrightness", 60))
HIGH_BRIGHTNESS = float(os.environ.get("state.highBrightness", 200))
GRAYSCALE_TOL = float(os.environ.get("state.grayscaleTol", 2))
EXPORT_CSV = os.environ.get("state.exportCsv", "true") == "true"


@my_app.callback("do")
@sly.timeit
def do(**kwargs):
    cfg = {
        "blur_th": BLUR_TH,
        "low_brightness": LOW_BRIGHTNESS,
        "high_brightness": HIGH_BRIGHTNESS,
        "grayscale_tol": GRAYSCALE_TOL,
    }

    rows = []
    datasets = (
        [DATASET_ID]
        if DATASET_ID
        else [d.id for d in api.dataset.get_list(PROJECT_ID)]
    )

    # ---- Counters for UI/log stats ----
    total = blur_cnt = dark_cnt = bright_cnt = gray_cnt = 0

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
            total += 1

            blur_cnt += int(res["blur"])
            dark_cnt += int(res["low_brightness"])
            bright_cnt += int(res["high_brightness"])
            gray_cnt += int(res["grayscale"])

            rows.append({"image": img_info.name, **res})

    # ---- SHOW STATS TO USER (this is your “UI”) ----
    sly.logger.info("Image Quality Summary", extra={
        "total_images": total,
        "blurred": blur_cnt,
        "too_dark": dark_cnt,
        "too_bright": bright_cnt,
        "grayscale": gray_cnt,
    })

    # ---- CSV OUTPUT ----
    if EXPORT_CSV and rows:
        df = pd.DataFrame(rows)
        csv_path = "/tmp/image_quality_report.csv"
        df.to_csv(csv_path, index=False)

        api.file.upload(
            TEAM_ID,
            csv_path,
            f"/image_quality_analyzer/{PROJECT_ID}_report.csv"
        )

    my_app.stop()


def main():
    my_app.run(initial_events=[{"command": "do"}])


if __name__ == "__main__":
    sly.main_wrapper("main", main)
