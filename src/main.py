import os
import tempfile
import cv2
import pandas as pd
import supervisely as sly

from quality import analyze_image


# -------------------------------------------------
# ENV & API
# -------------------------------------------------
api = sly.Api.from_env()

PROJECT_ID = sly.env.project_id()
DATASET_ID = sly.env.dataset_id(raise_not_found=False)
TEAM_ID = sly.env.team_id()

sly.logger.info(
    "App context",
    extra={
        "project_id": PROJECT_ID,
        "dataset_id": DATASET_ID,
    },
)

# -------------------------------------------------
# CREATE GUI APPLICATION
# -------------------------------------------------
app = sly.Application()


# -------------------------------------------------
# UI EVENT (called from modal.html button)
# -------------------------------------------------
@app.event("run")
def run(state, context):
    sly.logger.info("Run button clicked", extra={"state": state})

    # ---- read UI state ----
    blur_th = float(state.get("blurTh", 100))
    low_brightness = float(state.get("lowBrightness", 60))
    high_brightness = float(state.get("highBrightness", 200))
    grayscale_tol = float(state.get("grayscaleTol", 2))
    export_csv = state.get("exportCsv", True)

    cfg = {
        "blur_th": blur_th,
        "low_brightness": low_brightness,
        "high_brightness": high_brightness,
        "grayscale_tol": grayscale_tol,
    }

    # ---- datasets to process ----
    if DATASET_ID:
        dataset_ids = [DATASET_ID]
    else:
        dataset_ids = [d.id for d in api.dataset.get_list(PROJECT_ID)]

    rows = []
    total = blurred = dark = bright = gray = 0

    # ---- main processing ----
    for ds_id in dataset_ids:
        images = api.image.get_list(ds_id)

        for img_info in images:
            tmp_path = os.path.join(tempfile.gettempdir(), img_info.name)
            api.image.download(img_info.id, tmp_path)

            img = cv2.imread(tmp_path)
            if img is None:
                continue

            res = analyze_image(img, cfg)

            total += 1
            blurred += int(res["blur"])
            dark += int(res["low_brightness"])
            bright += int(res["high_brightness"])
            gray += int(res["grayscale"])

            rows.append(
                {
                    "image": img_info.name,
                    **res,
                }
            )

    # ---- logging summary ----
    sly.logger.info(
        "Image Quality Summary",
        extra={
            "total_images": total,
            "blurred": blurred,
            "too_dark": dark,
            "too_bright": bright,
            "grayscale": gray,
        },
    )

    # ---- export CSV ----
    if export_csv and rows:
        df = pd.DataFrame(rows)
        csv_local = "/sly-app-data/image_quality_report.csv"
        df.to_csv(csv_local, index=False)

        remote_path = f"/image_quality_analyzer/{PROJECT_ID}_report.csv"
        api.file.upload(
            team_id=TEAM_ID,
            local_path=csv_local,
            remote_path=remote_path,
        )

        sly.logger.info("CSV uploaded", extra={"path": remote_path})

    # ---- update UI state (IMPORTANT: keeps app alive) ----
    state["finished"] = True
    state["totalImages"] = total
    state["blurred"] = blurred
    state["tooDark"] = dark
    state["tooBright"] = bright
    state["grayscale"] = gray


# -------------------------------------------------
# DO NOT EXIT APP
# -------------------------------------------------
# No app.stop()
# No main()
# No auto-exit
