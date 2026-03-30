import os
os.environ["ORT_SKIP_OPSET_VALIDATION"] = "1"
os.environ["ORT_LOGGING_LEVEL"] = "3"

from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration


# -------------------- Streamlit --------------------
st.set_page_config(page_title="SafeGuard PRO", layout="centered")
st.title("SafeGuard PRO — контроль СИЗ")

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "best.onnx"
CLASSES_PATH = ROOT / "classes.txt"


# -------------------- TURN для Streamlit Cloud --------------------
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


# -------------------- Загрузка --------------------
@st.cache_resource
def load_classes():
    return [x.strip() for x in CLASSES_PATH.read_text().splitlines()]


@st.cache_resource
def load_model():
    sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    return sess, input_name


classes = load_classes()
sess, input_name = load_model()


# -------------------- Проверки классов --------------------
def is_person(label):
    return label == "person"


def is_helmet(label):
    return label == "helmet"


def is_vest(label):
    return label == "vest"


def is_no_helmet(label):
    return label == "no-helmet"


def is_no_vest(label):
    return label == "no-vest"


# -------------------- Инференс --------------------
def process_frame(img):
    h0, w0 = img.shape[:2]

    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = cv2.resize(x, (640, 640))
    x = np.transpose(x, (2, 0, 1))[None, ...]

    out = sess.run(None, {input_name: x})[0]

    if out.ndim == 3:
        out = out[0]

    people = []
    helmets = []
    vests = []
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

        elif is_helmet(label):
            helmets.append([x1, y1, x2, y2])

        elif is_vest(label):
            vests.append([x1, y1, x2, y2])

        elif is_no_helmet(label) or is_no_vest(label):
            no_boxes.append([x1, y1, x2, y2])

    safe = 0
    danger = 0

    for p in people:
        px1, py1, px2, py2 = p

        has_ppe = False
        has_no = False

        # Проверяем helmet/vest внутри человека
        for h in helmets:
            cx = (h[0] + h[2]) / 2
            cy = (h[1] + h[3]) / 2
            if px1 < cx < px2 and py1 < cy < py2:
                has_ppe = True

        for v in vests:
            cx = (v[0] + v[2]) / 2
            cy = (v[1] + v[3]) / 2
            if px1 < cx < px2 and py1 < cy < py2:
                has_ppe = True

        for nb in no_boxes:
            if not (nb[2] < px1 or nb[0] > px2 or nb[3] < py1 or nb[1] > py2):
                has_no = True

        if has_no:
            cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 3)
            danger += 1
        else:
            if has_ppe:
                cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 0), 3)
                safe += 1
            else:
                cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 3)
                danger += 1

    return img, safe, danger


# -------------------- LIVE --------------------
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed, _, _ = process_frame(img)
        return av.VideoFrame.from_ndarray(processed, format="bgr24")


tab1, tab2 = st.tabs(["LIVE", "Фото"])

with tab1:
    webrtc_streamer(
        key="live",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with tab2:
    uploaded = st.file_uploader("Загрузите фото", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = cv2.cvtColor(np.array(Image.open(uploaded).convert("RGB")), cv2.COLOR_RGB2BGR)
        result, safe, danger = process_frame(img)
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.write(f"SAFE: {safe} | DANGER: {danger}")
