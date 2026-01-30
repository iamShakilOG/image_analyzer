import os
import tempfile
import cv2
import pandas as pd
import supervisely as sly

from supervisely.app.widgets import (
    Container, Field, InputNumber, Button, Text, Table
)

from quality import analyze_image

api = sly.Api.from_env()

# -------- UI --------
blur_th = InputNumber(100, min=1, step=10)
low_brightness = InputNumber(60, min=0, max=255)
high_brightness = InputNumber(200, min=0, max=255)
grayscale_tol = InputNumber(2, min=0, max=10)

run_btn = Button("Run Analysis", button_size="large")

summary = Text("")
table = Table(columns=["Metric", "Count"], data=[])

layout = Container(widgets=[
    Field("Blur threshold", blur_th),
    Field("Low brightness", low_brightness),
    Field("High brightness", high_brightness),
    Field("Grayscale tolerance", grayscale_tol),
    run_btn,
    summary,
    table
])

# ✅ THIS IS THE KEY LINE
app = sly.Application(layout=layout)

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
            blur += int(res["blur"])
            dark += int(res["low_brightness"])
            bright += int(res["high_brightness"])
            gray += int(res["grayscale"])

            rows.append({"image": img_info.name, **res})

    summary.set(
        f"Processed: {total}\n"
        f"Blurred: {blur}\n"
        f"Too dark: {dark}\n"
        f"Too bright: {bright}\n"
        f"Grayscale: {gray}",
        status="success"
    )

    table.data = [
        ["Total images", total],
        ["Blurred", blur],
        ["Too dark", dark],
        ["Too bright", bright],
        ["Grayscale", gray],
    ]

    if rows:
        df = pd.DataFrame(rows)
        csv_path = "/tmp/image_quality_report.csv"
        df.to_csv(csv_path, index=False)

        api.file.upload(
            team_id=sly.env.team_id(),
            local_path=csv_path,
            remote_path=f"/image_quality_analyzer/{PROJECT_ID}_report.csv"
        )
