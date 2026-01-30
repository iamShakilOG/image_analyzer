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

# -----------------------------------
# ENV + API
# -----------------------------------
api = sly.Api.from_env()

PROJECT_ID = sly.env.project_id()
DATASET_ID = sly.env.dataset_id(raise_not_found=False)

# -----------------------------------
# UI WIDGETS (NO LABEL ARGUMENTS ❗)
# -----------------------------------
blur_input = InputNumber(value=100, min=1)
low_brightness_input = InputNumber(value=60, min=0)
high_brightness_input = InputNumber(value=200, min=0)
grayscale_tol_input = InputNumber(value=2, min=0)
export_csv_checkbox = Checkbox("Export CSV", checked=True)

run_button = Button("Run analysis", button_type="primary")

status_text = Text()
progress = SlyTqdm()
progress.hide()

# -----------------------------------
# UI LAYOUT
# -----------------------------------
settings_card = Card(
    title="Image Quality Settings",
    content=Container([
        Text("Blur threshold"),
        blur_input,

        Text("Low brightness threshold"),
        low_brightness_input,

        Text("High brightness threshold"),
        high_brightness_input,

        Text("Grayscale tolerance"),
        grayscale_tol_input,

        export_csv_checkbox,
    ]),
)

output_card = Card(
    title="Output",
    content=Container([
        run_button,
        progress,
        status_text,
    ]),
)

layout = Container([settings_card, output_card])

app = sly.Application(layout=layout)

# -----------------------------------
# BUTTON CALLBACK (THIS IS THE KEY)
# -----------------------------------
@run_button.click
def run():
    status_text.text = "Running image quality analysis..."
    progress.show()

    cfg = {
        "blur_th": blur_input.get_value(),
        "low_brightness": low_brightness_input.get_value(),
        "high_brightness": high_brightness_input.get_value(),
        "grayscale_tol": grayscale_tol_input.get_value(),
    }

    if DATASET_ID is not None:
        dataset_ids = [DATASET_ID]
    else:
        dataset_ids = [d.id for d in api.dataset.get_list(PROJECT_ID)]

    rows = []
    stats = {"total": 0, "blur": 0, "dark": 0, "bright": 0, "gray": 0}

    images = []
    for ds_id in dataset_ids:
        images.extend(api.image.get_list(ds_id))

    progress.set_total(len(images))

    for img_info in images:
        tmp = os.path.join(tempfile.gettempdir(), img_info.name)
        api.image.download(img_info.id, tmp)

        img = cv2.imread(tmp)
        if img is None:
            progress.update(1)
            continue

        res = analyze_image(img, cfg)

        stats["total"] += 1
        stats["blur"] += int(res["blur"])
        stats["dark"] += int(res["low_brightness"])
        stats["bright"] += int(res["high_brightness"])
        stats["gray"] += int(res["grayscale"])

        rows.append({"image": img_info.name, **res})
        progress.update(1)

    progress.hide()

    # -----------------------------------
    # CSV EXPORT
    # -----------------------------------
    if export_csv_checkbox.is_checked() and rows:
        df = pd.DataFrame(rows)
        csv_path = "/sly-app-data/image_quality_report.csv"
        df.to_csv(csv_path, index=False)

        api.file.upload(
            team_id=sly.env.team_id(),
            local_path=csv_path,
            remote_path=f"/image_quality_analyzer/{PROJECT_ID}_report.csv",
        )

    # -----------------------------------
    # UI OUTPUT
    # -----------------------------------
    status_text.text = (
        f"✅ Done!\n\n"
        f"Images analyzed: {stats['total']}\n"
        f"Blurred: {stats['blur']}\n"
        f"Too dark: {stats['dark']}\n"
        f"Too bright: {stats['bright']}\n"
        f"Grayscale: {stats['gray']}"
    )

    sly.logger.info("Image Quality Summary", extra=stats)
