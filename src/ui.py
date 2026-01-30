import supervisely_lib as sly

blur_th = sly.number("Blur threshold (lower = more blurry)", 100, min=1, step=10)
low_brightness = sly.number("Low brightness threshold", 60, min=0, max=255)
high_brightness = sly.number("High brightness threshold", 200, min=0, max=255)
grayscale_tol = sly.number("Grayscale tolerance", 2, min=0, max=10)
export_csv = sly.checkbox("Export CSV report", True)
