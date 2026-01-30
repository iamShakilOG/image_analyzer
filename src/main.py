import os
import tempfile
import cv2
import pandas as pd
import supervisely as sly

from ui import layout, run_btn, blur_th, low_brightness, high_brightness, grayscale_tol, export_csv
from quality import analyze_image

api = sly.Api.from_env()
app = sly.App(layout=layout)

@run_btn.click
def run():
    project_id = sly.env.project_id()
    dataset_id = sly.env.dataset_id(raise_not_found=False)
    team_id = sly.env.team_id()

    # ---- Ensure tag metas ----
    meta_json = api.project.get_meta(project_id)
    meta = sly.ProjectMeta.from_json(meta_json)

    tag_names = ["blur", "low_brightness", "high_brightness", "grayscale"]
    existing = {t.name for t in meta.tag_metas}
    new_metas = [
        sly.TagMeta(name, sly.TagValueType.NONE)
        for name in tag_names if name not in existing
    ]

    if new_metas:
        meta = meta.add_tag_metas(new_metas)
        api.project.update_meta(project_id, meta)

    cfg = {
        "blur_th": blur_th.get_value(),
        "low_brightness": low_brightness.get_value(),
        "high_brightness": high_brightness.get_value(),
        "grayscale_tol": grayscale_tol.get_value()
    }

    rows = []
    datasets = [dataset_id] if dataset_id else [d.id for d in api.dataset.get_list(project_id)]

    for ds_id in datasets:
        for img_info in api.image.get_list(ds_id):
            local_path = os.path.join(tempfile.gettempdir(), img_info.name)
            api.image.download(img_info.id, local_path)

            img = cv2.imread(local_path)
            if img is None:
                continue

            res = analyze_image(img, cfg)

            ann = api.annotation.download(img_info.id)
            tags = [
                sly.Tag(meta.get_tag_meta(k))
                for k in tag_names if res[k]
            ]
            ann = ann.add_tags(tags)
            api.annotation.upload_ann(img_info.id, ann)

            rows.append({"image": img_info.name, **res})

    if export_csv.is_checked() and rows:
        df = pd.DataFrame(rows)
        csv_path = "/tmp/image_quality_report.csv"
        df.to_csv(csv_path, index=False)
        api.file.upload(team_id, csv_path, f"/image_quality_reports/{project_id}_report.csv")
