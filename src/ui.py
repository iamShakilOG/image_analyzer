from supervisely.app.widgets import Container, Field, InputNumber, Checkbox, Button

blur_th = InputNumber(100, min=1, step=10)
low_brightness = InputNumber(60, min=0, max=255)
high_brightness = InputNumber(200, min=0, max=255)
grayscale_tol = InputNumber(2, min=0, max=10)
export_csv = Checkbox("Export CSV report")
run_btn = Button("Run Analysis", button_size="large")

layout = Container([
    Field("Blur threshold (lower = more blurry)", blur_th),
    Field("Low brightness threshold", low_brightness),
    Field("High brightness threshold", high_brightness),
    Field("Grayscale tolerance", grayscale_tol),
    export_csv,
    run_btn
])