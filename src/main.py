import supervisely as sly
import cv2
import pandas as pd
from ui import layout, run_btn, blur_th, low_brightness, high_brightness, grayscale_tol, export_csv
from quality import analyze_image

api = sly.Api.from_env()
app = sly.App(layout=layout)

@run_btn.click
def run():
    project_id = sly.env.project_id()
    dataset_id = sly.env.dataset_id(raise_not_found=False)

    cfg = {
        "blur_th": blur_th.get_value(),
        "low_brightness": low_brightness.get_value(),
        "high_brightness": high_brightness.get_value(),
        "grayscale_tol": grayscale_tol.get_value()
    }

    rows = []
    datasets = [dataset_id] if dataset_id else [d.id for d in api.dataset.get_list(project_id)]

    for ds_id in datasets:
        images = api.image.get_list(ds_id)
        for img_info in images:
            local_path = api.image.download_path(img_info.id)
            img = cv2.imread(local_path)
            res = analyze_image(img, cfg)

            ann = api.annotation.download(img_info.id)
            tags = [sly.Tag(k) for k in ["blur","low_brightness","high_brightness","grayscale"] if res[k]]
            ann = ann.add_tags(tags)
            api.annotation.upload_ann(img_info.id, ann)

            rows.append({"image": img_info.name, **res})

    if export_csv.is_checked():
        df = pd.DataFrame(rows)
        csv_path = "/tmp/image_quality_report.csv"
        df.to_csv(csv_path, index=False)
        api.file.upload(csv_path, project_id)