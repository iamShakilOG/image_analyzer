import os
import tempfile
import cv2
import pandas as pd

import supervisely as sly
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    InputNumber,
    Checkbox,
    Text,
    SlyTqdm,
)

from quality import analyze_image


# --------------------------------------------------
# ENV & API
# --------------------------------------------------
api = sly.Api.from_env()

PROJECT_ID = sly.env.project_id()
DATASET_ID = sly.env.dataset_id(raise_not_found=False)
TEAM_ID = sly.env.team_id()

if PROJECT_ID is None:
    raise RuntimeError("Run app from Project or Dataset context")


# --------------------------------------------------
# UI WIDGETS
# --------------------------------------------------
blur_input = InputNumber(value=100, min=1, label="Blur threshold")
low_brightness_input = InputNumber(value=60, min=0, label="Low brightness")
high_brightness_input = InputNumber(value=200, min=0, label="High brightness")
grayscale_tol_input = InputNumber(value=2, min=0, label="Grayscale tolerance")
export_csv_checkbox = Checkbox("Export CSV", checked=True)

run_btn = Button("Run analysis", button_type="primary")
progress = SlyTqdm()
progress.hide()

stats_text = Text()
stats_text.hide()


settings_card = Card(
    title="Settings",
    content=Container(
        [
            blur_input,
            low_brightness_input,
            high_brightness_input,
            grayscale_tol_input,
            export_csv_checkbox,
            run_btn,
            progress,
            stats_text,
        ]
    ),
)

app = sly.Application(layout=settings_card)


# --------------------------------------------------
# BUTTON HANDLER
# --------------------------------------------------
@run_btn.click
def run_analysis():
    run_btn.disable()
    progress.show()
    stats_text.hide()

    cfg = {
        "blur_th": blur_input.get_value(),
        "low_brightness": low_brightness_input.get_value(),
        "high_brightness": high_brightness_input.get_value(),
        "grayscale_tol": grayscale_tol_input.get_value(),
    }

    # decide datasets
    if DATASET_ID is not None:
        dataset_ids = [DATASET_ID]
    else:
        dataset_ids = [d.id for d in api.dataset.get_list(PROJECT_ID)]

    rows = []
    stats = {"total": 0, "blur": 0, "dark": 0, "bright": 0, "gray": 0}

    images = []
    for ds_id in dataset_ids:
        images.extend(api.image.get_list(ds_id))

    with progress(total=len(images)) as pbar:
        for img_info in images:
            tmp_path = os.path.join(tempfile.gettempdir(), img_info.name)
            api.image.download(img_info.id, tmp_path)

            img = cv2.imread(tmp_path)
            if img is None:
                pbar.update(1)
                continue

            res = analyze_image(img, cfg)

            stats["total"] += 1
            stats["blur"] += int(res["blur"])
            stats["dark"] += int(res["low_brightness"])
            stats["bright"] += int(res["high_brightness"])
            stats["gray"] += int(res["grayscale"])

            rows.append({"image": img_info.name, **res})
            pbar.update(1)

    # Export CSV
    if export_csv_checkbox.is_checked() and rows:
        df = pd.DataFrame(rows)
        local_csv = "/sly-app-data/image_quality_report.csv"
        df.to_csv(local_csv, index=False)

        remote_path = f"/image_quality_analyzer/{PROJECT_ID}_report.csv"
        api.file.upload(TEAM_ID, local_csv, remote_path)

    stats_text.set(
        text=(
            f"Images analyzed: {stats['total']}\n"
            f"Blurred: {stats['blur']}\n"
            f"Too dark: {stats['dark']}\n"
            f"Too bright: {stats['bright']}\n"
            f"Grayscale: {stats['gray']}"
        ),
        status="success",
    )

    stats_text.show()
    run_btn.enable()
