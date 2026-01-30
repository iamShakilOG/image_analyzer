from supervisely.app.widgets import (
    Container, Field, InputNumber, Button,
    Text, Table
)

blur_th = InputNumber(100, min=1, step=10)
low_brightness = InputNumber(60, min=0, max=255)
high_brightness = InputNumber(200, min=0, max=255)
grayscale_tol = InputNumber(2, min=0, max=10)

run_btn = Button("Run Analysis", button_size="large")

summary_text = Text("", status="info")

table = Table(
    columns=["Metric", "Count"],
    data=[]
)

controls = Container([
    Field("Blur threshold", blur_th),
    Field("Low brightness", low_brightness),
    Field("High brightness", high_brightness),
    Field("Grayscale tolerance", grayscale_tol),
    run_btn
])

layout = Container([
    controls,
    summary_text,
    table
])
