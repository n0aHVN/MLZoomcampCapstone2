from ultralytics import YOLO

HOME = '.'
model = YOLO("yolo11n.pt")
search_space = {
    "lr0": (0.01, 0.001),
    "lrf": (0.1,0.5),
    "degrees": (0.0, 45.0),
    "scale": (0.0, 0.5,0.9),
    "shear": (0.0,5.0)
}
model.tune(
    data="./datasets/dataset.yaml",
    epochs = 30,
    optimizer="AdamW",
    space = search_space,
    save=True,
    val = True,
    batch = 16,
    patience = 8,
    device = "0",
    amp=False
)

