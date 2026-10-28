from ultralytics import YOLO

MODEL_PATH = "../models/best.pt"

print("Loading YOLO...")

model = YOLO(MODEL_PATH)

print("YOLO loaded successfully!")
print("Classes:", model.names)

IMAGE_PATH = r"C:\Users\Syed\Downloads\fundus2.jpg"

print("\nRunning prediction...")

results = model.predict(
    source=IMAGE_PATH,
    imgsz=640,
    conf=0.25,
    save=True
)

print("\nPrediction completed!")

for result in results:

    print("\nDetected lesions:")

    if result.boxes is None or len(result.boxes) == 0:
        print("No lesions detected.")
        continue

    for box in result.boxes:

        class_id = int(
            box.cls[0].item()
        )

        confidence = float(
            box.conf[0].item()
        )

        class_name = model.names[
            class_id
        ]

        print(
            f"{class_name}: "
            f"{confidence * 100:.2f}%"
        )

print("\nAnnotated image saved by YOLO.")