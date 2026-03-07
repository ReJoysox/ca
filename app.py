import os
os.environ["ORT_SKIP_OPSET_VALIDATION"] = "1"
os.environ["ORT_LOGGING_LEVEL"] = "3"

from pathlib import Path
import threading
import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image
import av

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
)

# ----------------- Streamlit must be first -----------------
st.set_page_config(page_title="SafeGuard PRO", layout="centered")
st.title("SafeGuard PRO — LIVE контроль СИЗ")

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "best.onnx"
CLASSES_PATH = ROOT / "classes.txt"


# ✅ TURN чтобы LIVE работал на Streamlit Cloud
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": [
                    "turn:openrelay.metered.ca:80?transport=tcp",
                    "turn:openrelay.metered.ca:443?transport=tcp",
                    "turns:openrelay.metered.ca:443?transport=tcp",
                ],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ]
    }
)


# ---------------- Model ----------------
@st.cache_resource
def load_classes():
    return [x.strip() for x in CLASSES_PATH.read_text().splitlines()]


@st.cache_resource
def load_session():
    sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    return sess, input_name


classes = load_classes()
sess, input_name = load_session()


def norm(s):
    return s.lower().replace("_", "-")


def is_person(label):
    l = norm(label)
    return "person" in l


def is_no(label):
    l = norm(label)
    return "no-helmet" in l or "no-vest" in l


def is_ppe(label):
    l = norm(label)
    if "no-" in l:
        return False
    return "helmet" in l or "vest" in l


def process_frame(img):
    h0, w0 = img.shape[:2]

    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = cv2.resize(x, (640, 640))
    x = np.transpose(x, (2, 0, 1))[None, ...]

    out = sess.run(None, {input_name: x})[0]

    if out.ndim == 3:
        out = out[0]

    safe = 0
    danger = 0

    people = []
    ppe_boxes = []
    no_boxes = []

    for det in out:
        x1, y1, x2, y2, score, cls_id = det
        if score < 0.25:
            continue

        label = classes[int(cls_id)]
        x1 = int(x1 * w0 / 640)
        x2 = int(x2 * w0 / 640)
        y1 = int(y1 * h0 / 640)
        y2 = int(y2 * h0 / 640)

        if is_person(label):
            people.append([x1, y1, x2, y2])
        elif is_ppe(label):
            ppe_boxes.append([x1, y1, x2, y2])
        elif is_no(label):
            no_boxes.append([x1, y1, x2, y2])

    for p in people:
        px1, py1, px2, py2 = p

        has_ppe = any(
            (px1 < (b[0]+b[2])/2 < px2) and
            (py1 < (b[1]+b[3])/2 < py2)
            for b in ppe_boxes
        )

        has_no = any(
            not (b[2] < px1 or b[0] > px2 or b[3] < py1 or b[1] > py2)
            for b in no_boxes
        )

        if has_no:
            cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 3)
            danger += 1
        elif has_ppe:
            cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 0), 3)
            safe += 1
        else:
            cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 3)
            danger += 1

    return img, safe, danger


class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed, _, _ = process_frame(img)
        return av.VideoFrame.from_ndarray(processed, format="bgr24")


webrtc_streamer(
    key="live",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
