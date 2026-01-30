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

api = sly.Api.from_env()
PROJECT_ID = sly.env.project_id()
DATASET_ID = sly.env.dataset_id(raise_not_found=False)
TEAM_ID = sly.env.team_id()

# ---------------- UI ----------------
blur = InputNumber(value=100, min=1)
low_b = InputNumber(value=60, min=0)
high_b = InputNumber(value=200, min=0)
gray = InputNumber(value=2, min=0)
export_csv = Checkbox("Export CSV", checked=True)

run_btn = Button("Run", button_type="primary")
status = Text()
progress = SlyTqdm()
progress.hide()

@run_btn.click
def run():
    status.text = "Running analysis..."
    progress.show()

    cfg = {
        "blur_th": blur.get_value(),
        "low_brightness": low_b.get_value(),
        "high_brightness": high_b.get_value(),
        "grayscale_tol": gray.get_value(),
    }

    datasets = [DATASET_ID] if DATASET_ID else [
        d.id for d in api.dataset.get_list(PROJECT_ID)
    ]

    rows = []
    images = []
    for ds in datasets:
        images.extend(api.image.get_list(ds))

    progress.set_total(len(images))

    for img in images:
        tmp = os.path.join(tempfile.gettempdir(), img.name)
        api.image.download(img.id, tmp)

        im = cv2.imread(tmp)
        if im is None:
            progress.update(1)
            continue

        res = analyze_image(im, cfg)
        rows.append({"image": img.name, **res})
        progress.update(1)

    progress.hide()

    if export_csv.is_checked() and rows:
        df = pd.DataFrame(rows)
        local = "/sly-app-data/report.csv"
        df.to_csv(local, index=False)
        api.file.upload(TEAM_ID, local, f"/image_quality/{PROJECT_ID}.csv")

    status.text = f"✅ Done. Images analyzed: {len(rows)}"

layout = Container([
    Card("Settings", content=Container([
        Text("Blur threshold"), blur,
        Text("Low brightness"), low_b,
        Text("High brightness"), high_b,
        Text("Grayscale tolerance"), gray,
        export_csv,
    ])),
    Card("Run", content=Container([run_btn, progress, status])),
])

app = sly.Application(layout=layout)
