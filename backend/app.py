import os
import base64
import requests
from dotenv import load_dotenv
from ultralytics import YOLO

import cv2
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf

from tensorflow import keras

from torchvision import models

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response


load_dotenv(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".env"
    )
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "resnet_dr_best.pth"
)

UNET_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_retinal_unet.keras"
)

YOLO_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best.pt"
)

DEVICE = torch.device("cpu")

CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy"
]


# --------------------------------------------------
# Create ResNet-50
# --------------------------------------------------

print("Loading ResNet-50...")

model = models.resnet50(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    5
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

print("ResNet-50 loaded successfully.")

# --------------------------------------------------
# U-Net custom functions
# --------------------------------------------------

def dice_coefficient(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    smooth = 1e-6

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(
        y_true * y_pred
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        tf.reduce_sum(y_true)
        + tf.reduce_sum(y_pred)
        + smooth
    )

    return dice


def dice_loss(y_true, y_pred):

    return 1.0 - dice_coefficient(
        y_true,
        y_pred
    )


bce = tf.keras.losses.BinaryCrossentropy()


def combined_loss(y_true, y_pred):

    return bce(
        y_true,
        y_pred
    ) + dice_loss(
        y_true,
        y_pred
    )


# --------------------------------------------------
# Load U-Net
# --------------------------------------------------

print("Loading U-Net...")

unet_model = keras.models.load_model(
    UNET_MODEL_PATH,
    custom_objects={
        "combined_loss": combined_loss,
        "dice_coefficient": dice_coefficient,
        "dice_loss": dice_loss
    },
    compile=False
)

print("U-Net loaded successfully.")

# --------------------------------------------------
# Load YOLO
# --------------------------------------------------

print("Loading YOLO...")

yolo_model = YOLO(
    YOLO_MODEL_PATH
)

print("YOLO loaded successfully.")

print(
    "YOLO classes:",
    yolo_model.names
)

# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Retinal AI Backend",
    description="AI backend for retinal image analysis"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def preprocess_image(image_bytes):

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    img = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if img is None:
        raise ValueError("Could not read the uploaded image.")

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = cv2.resize(
        img,
        (224, 224)
    )

    # CLAHE
    lab = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    img = cv2.cvtColor(
        cv2.merge((l, a, b)),
        cv2.COLOR_LAB2RGB
    )

    # Normalize
    img = img.astype(np.float32) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    img = (img - mean) / std

    # HWC -> CHW
    img = np.transpose(
        img,
        (2, 0, 1)
    )

    # Add batch dimension
    img = np.expand_dims(
        img,
        axis=0
    )

    tensor = torch.from_numpy(
        img
    ).float()

    return tensor.to(DEVICE)

# --------------------------------------------------
# U-Net full-image prediction
# --------------------------------------------------

def predict_unet_full_image(
    image,
    patch_size=128,
    stride=64,
    threshold=0.5
):

    # Same normalization used during training
    image_float = (
        image.astype(np.float32) / 255.0
    )

    h, w = image_float.shape[:2]

    prediction_sum = np.zeros(
        (h, w),
        dtype=np.float32
    )

    prediction_count = np.zeros(
        (h, w),
        dtype=np.float32
    )

    # Same patch positioning strategy
    y_positions = list(
        range(
            0,
            h - patch_size + 1,
            stride
        )
    )

    x_positions = list(
        range(
            0,
            w - patch_size + 1,
            stride
        )
    )

    # Handle images whose dimensions aren't
    # perfectly divisible by the stride
    if y_positions[-1] != h - patch_size:
        y_positions.append(
            h - patch_size
        )

    if x_positions[-1] != w - patch_size:
        x_positions.append(
            w - patch_size
        )

    patches = []
    locations = []

    for y in y_positions:

        for x in x_positions:

            patch = image_float[
                y:y + patch_size,
                x:x + patch_size
            ]

            patches.append(patch)
            locations.append((y, x))

    patches = np.array(
        patches,
        dtype=np.float32
    )

    # Predict all patches
    predictions = unet_model.predict(
        patches,
        batch_size=8,
        verbose=0
    )

    # Reconstruct full image
    for prediction, (y, x) in zip(
        predictions,
        locations
    ):

        prediction = prediction[:, :, 0]

        prediction_sum[
            y:y + patch_size,
            x:x + patch_size
        ] += prediction

        prediction_count[
            y:y + patch_size,
            x:x + patch_size
        ] += 1

    # Average overlapping predictions
    probability_map = (
        prediction_sum /
        np.maximum(
            prediction_count,
            1
        )
    )

    # Binary vessel mask
    binary_mask = (
        probability_map >= threshold
    ).astype(np.uint8)

    return probability_map, binary_mask

# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "Retinal AI Backend is running",
        "model": "ResNet-50",
        "status": "ready"
    }


# --------------------------------------------------
# ResNet prediction
# --------------------------------------------------

@app.post("/predict/resnet")
async def predict_resnet(
    file: UploadFile = File(...)
):

    try:

        image_bytes = await file.read()

        image_tensor = preprocess_image(
            image_bytes
        )

        with torch.no_grad():

            output = model(
                image_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = probabilities[
                0,
                predicted_class
            ].item()

        return {
            "success": True,
            "model": "ResNet-50",
            "prediction": CLASS_NAMES[predicted_class],
            "class_id": predicted_class,
            "confidence": round(
                confidence * 100,
                2
            ),
            "probabilities": {
                CLASS_NAMES[i]: round(
                    probabilities[0, i].item() * 100,
                    2
                )
                for i in range(5)
            }
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }

# --------------------------------------------------
# U-Net vessel segmentation
# --------------------------------------------------

@app.post("/predict/unet")
async def predict_unet(
    file: UploadFile = File(...)
):

    try:

        # ----------------------------------------------
        # Read uploaded image
        # ----------------------------------------------

        image_bytes = await file.read()

        image_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise ValueError(
                "Could not read the uploaded image."
            )

        # BGR -> RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------
        # U-Net prediction
        # ----------------------------------------------

        probability_map, binary_mask = (
            predict_unet_full_image(image)
        )

        # ----------------------------------------------
        # Create black/white vessel mask
        # ----------------------------------------------

        mask_image = (
            binary_mask * 255
        ).astype(np.uint8)

        # ----------------------------------------------
        # Create vessel overlay
        # ----------------------------------------------

        overlay = image.copy()

        # Highlight predicted vessels
        overlay[binary_mask == 1] = [0, 255, 0]

        # Blend original + highlighted vessels
        overlay_image = cv2.addWeighted(
            image,
            0.65,
            overlay,
            0.35,
            0
        )

        # ----------------------------------------------
        # Encode mask as PNG
        # ----------------------------------------------

        mask_success, encoded_mask = cv2.imencode(
            ".png",
            mask_image
        )

        if not mask_success:
            raise ValueError(
                "Could not encode segmentation mask."
            )

        # ----------------------------------------------
        # Encode overlay as PNG
        # ----------------------------------------------

        overlay_success, encoded_overlay = cv2.imencode(
            ".png",
            cv2.cvtColor(
                overlay_image,
                cv2.COLOR_RGB2BGR
            )
        )

        if not overlay_success:
            raise ValueError(
                "Could not encode overlay image."
            )

        # ----------------------------------------------
        # Convert images to Base64
        # ----------------------------------------------

        mask_base64 = base64.b64encode(
            encoded_mask.tobytes()
        ).decode("utf-8")

        overlay_base64 = base64.b64encode(
            encoded_overlay.tobytes()
        ).decode("utf-8")

        # ----------------------------------------------
        # Return results
        # ----------------------------------------------

        return {
            "success": True,
            "model": "U-Net",
            "message": "Retinal vessel segmentation completed.",
            "mask_image": (
                "data:image/png;base64,"
                + mask_base64
            ),
            "overlay_image": (
                "data:image/png;base64,"
                + overlay_base64
            )
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }

# --------------------------------------------------
# YOLO retinal lesion detection
# --------------------------------------------------

@app.post("/predict/yolo")
async def predict_yolo(
    file: UploadFile = File(...)
):

    try:

        # ------------------------------------------
        # Read uploaded image
        # ------------------------------------------

        image_bytes = await file.read()

        image_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise ValueError(
                "Could not read the uploaded image."
            )

        # ------------------------------------------
        # Run YOLO
        # ------------------------------------------

        results = yolo_model.predict(
            source=image,
            imgsz=640,
            conf=0.25,
            verbose=False
        )

        result = results[0]

        # ------------------------------------------
        # Extract detections
        # ------------------------------------------

        detections = []

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                class_name = yolo_model.names[
                    class_id
                ]

                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                )

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(
                        confidence * 100,
                        2
                    ),
                    "box": [
                        round(value, 2)
                        for value in coordinates
                    ]
                })

        # ------------------------------------------
        # Create annotated image
        # ------------------------------------------

        annotated_image = result.plot()

        success, encoded_image = cv2.imencode(
            ".jpg",
            annotated_image
        )

        if not success:
            raise ValueError(
                "Could not encode YOLO result."
            )

        annotated_base64 = base64.b64encode(
            encoded_image.tobytes()
        ).decode("utf-8")

        # ------------------------------------------
        # Summary
        # ------------------------------------------

        class_counts = {}

        for detection in detections:

            name = detection["class_name"]

            class_counts[name] = (
                class_counts.get(name, 0) + 1
            )

        # ------------------------------------------
        # Return result
        # ------------------------------------------

        return {
            "success": True,
            "model": "YOLO",
            "detections": detections,
            "total_detections": len(
                detections
            ),
            "class_counts": class_counts,
            "annotated_image": (
                "data:image/jpeg;base64,"
                + annotated_base64
            )
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }

    # --------------------------------------------------
# Email PDF report using Brevo
# --------------------------------------------------

@app.post("/report/email")
async def send_report_email(
    recipient_email: str = Form(...),
    patient_name: str = Form(...),
    patient_age: int = Form(...),
    patient_contact: str = Form(...),
    report_pdf: UploadFile = File(...)
):
    api_key = os.getenv("BREVO_API_KEY")
    sender_email = os.getenv("BREVO_SENDER_EMAIL")
    sender_name = os.getenv("BREVO_SENDER_NAME", "Retinal AI")

    if not api_key or not sender_email:
        raise HTTPException(
            status_code=500,
            detail="Email service is not configured."
        )

    if not report_pdf.filename or not report_pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please provide a PDF report."
        )

    pdf_bytes = await report_pdf.read()

    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid PDF."
        )

    pdf_base64 = base64.b64encode(
        pdf_bytes
    ).decode("utf-8")

    payload = {
        "sender": {
            "name": sender_name,
            "email": sender_email
        },
        "to": [
            {
                "email": recipient_email,
                "name": patient_name
            }
        ],
        "subject": "Your Retinal AI Analysis Report",
        "textContent": (
            f"Hello {patient_name},\n\n"
            "Please find your Retinal AI analysis report "
            "attached to this email.\n\n"
            "This report is generated by an AI research prototype "
            "and is not a medical diagnosis. Please consult a "
            "qualified eye-care professional for interpretation."
        ),
        "attachment": [
            {
                "name": "Retinal_AI_Report.pdf",
                "content": pdf_base64
            }
        ]
    }

    headers = {
        "accept": "application/json",
        "api-key": api_key,
        "content-type": "application/json"
    }

    try:
        response = requests.post(
            "https://api.brevo.com/v3/smtp/email",
            json=payload,
            headers=headers,
            timeout=30
        )

        if response.status_code not in (200, 201, 202):
            print(
                "Brevo email error:",
                response.status_code,
                response.text
            )

            raise HTTPException(
                status_code=502,
                detail="The email service could not send the report."
            )

        return {
            "success": True,
            "message": "Report sent successfully.",
            "recipient": recipient_email
        }

    except requests.RequestException:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to the email service."
        )