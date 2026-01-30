import os
import tempfile
import cv2
import pandas as pd
import supervisely as sly

from ui import (
    layout, run_btn, summary_text, table,
    blur_th, low_brightness, high_brightness, grayscale_tol
)
from quality import analyze_image

api = sly.Api.from_env()
app = sly.App(layout=layout)

PROJECT_ID = sly.env.project_id()
DATASET_ID = sly.env.dataset_id(raise_not_found=False)

@run_btn.click
def run():
    cfg = {
        "blur_th": blur_th.get_value(),
        "low_brightness": low_brightness.get_value(),
        "high_brightness": high_brightness.get_value(),
        "grayscale_tol": grayscale_tol.get_value(),
    }

    datasets = (
        [DATASET_ID]
        if DATASET_ID
        else [d.id for d in api.dataset.get_list(PROJECT_ID)]
    )

    total = blur = dark = bright = gray = 0
    rows = []

    for ds_id in datasets:
        for img_info in api.image.get_list(ds_id):
            tmp = os.path.join(tempfile.gettempdir(), img_info.name)
            api.image.download(img_info.id, tmp)

            img = cv2.imread(tmp)
            if img is None:
                continue

            res = analyze_image(img, cfg)
            total += 1
            blur += res["blur"]
            dark += res["low_brightness"]
            bright += res["high_brightness"]
            gray += res["grayscale"]

            rows.append({"image": img_info.name, **res})

    # ---- UPDATE UI ----
    summary_text.text = (
        f"Processed {total} images\n"
        f"Blurred: {blur}\n"
        f"Too dark: {dark}\n"
        f"Too bright: {bright}\n"
        f"Grayscale: {gray}"
    )

    table.data = [
        ["Total images", total],
        ["Blurred", blur],
        ["Too dark", dark],
        ["Too bright", bright],
        ["Grayscale", gray],
    ]

    # ---- EXPORT CSV ----
    if rows:
        df = pd.DataFrame(rows)
        csv_path = "/tmp/image_quality_report.csv"
        df.to_csv(csv_path, index=False)

        api.file.upload(
            team_id=sly.env.team_id(),
            local_path=csv_path,
            remote_path=f"/image_quality_analyzer/{PROJECT_ID}_report.csv"
        )

        sly.logger.info("CSV uploaded")

