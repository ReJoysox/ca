import os
# Важно: до импортов ultralytics/onnxruntime
os.environ["ORT_SKIP_OPSET_VALIDATION"] = "1"
os.environ["ORT_LOGGING_LEVEL"] = "3"

import threading
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration


# -------------------- Streamlit (первый st.*) --------------------
st.set_page_config(page_title="SafeGuard PRO", layout="centered")
st.title("🛡️ SafeGuard PRO — контроль каски (зелёный/красный)")


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "best.onnx"
CLASSES_PATH = ROOT / "classes.txt"

IMGSZ = 640


# -------------------- WebRTC (TURN чтобы работало на Streamlit Cloud) --------------------
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


# -------------------- Shared config for video thread --------------------
_lock = threading.Lock()
CFG = {"conf": 0.25, "person_conf": 0.12}  # общий порог и порог person отдельно


def cfg_set(**kwargs):
    with _lock:
        CFG.update(kwargs)


def cfg_get():
    with _lock:
        return dict(CFG)


@st.cache_resource
def load_classes():
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(f"Не найден classes.txt: {CLASSES_PATH}")
    return [x.strip() for x in CLASSES_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Не найден best.onnx: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH), task="detect")

    # Жёстко фиксируем имена классов по твоему classes.txt (чтобы ничего не путалось)
    names = load_classes()  # ["person","helmet","vest","no-helmet","no-vest"]
    model.names = {i: n for i, n in enumerate(names)}

    return model


def expand_person_up(p, img_h, ratio=0.65):
    """Person bbox часто обрезает голову -> расширяем вверх, чтобы каска попадала внутрь."""
    x1, y1, x2, y2 = p
    h = max(1.0, y2 - y1)
    y1 = max(0.0, y1 - ratio * h)
    y2 = min(float(img_h - 1), y2)
    return [x1, y1, x2, y2]


def inside_ratio(person_box, obj_box) -> float:
    """Доля площади объекта (каски) внутри бокса человека."""
    px1, py1, px2, py2 = person_box
    ox1, oy1, ox2, oy2 = obj_box

    ix1 = max(px1, ox1)
    iy1 = max(py1, oy1)
    ix2 = min(px2, ox2)
    iy2 = min(py2, oy2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    obj_area = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1) + 1e-9
    return inter / obj_area


def process_frame_helmet_only(img_bgr, model, conf, person_conf):
    # Чтобы person не отрезало, в predict ставим минимальный порог
    pred_conf = min(conf, person_conf, 0.10)

    res = model.predict(img_bgr, conf=pred_conf, imgsz=IMGSZ, iou=0.3, verbose=False)[0]
    boxes = res.boxes

    people = []      # list of xyxy
    helmets = []     # list of xyxy
    no_helmets = []  # list of xyxy (если модель даёт)

    for b in boxes:
        cls_id = int(b.cls[0])
        label = model.names.get(cls_id, str(cls_id))
        score = float(b.conf[0])
        xyxy = b.xyxy[0].tolist()

        if label == "person" and score >= person_conf:
            people.append(xyxy)
        elif label == "helmet" and score >= conf:
            helmets.append(xyxy)
        elif label == "no-helmet" and score >= conf:
            no_helmets.append(xyxy)

    safe = 0
    danger = 0
    img_h = img_bgr.shape[0]

    # Порог привязки каски к человеку (мягкий)
    HELMET_INSIDE = 0.03

    for p in people:
        px1, py1, px2, py2 = p
        p_exp = expand_person_up(p, img_h, ratio=0.65)

        # Каска считается "на человеке", если хоть немного её площади внутри расширенного person
        has_helmet = any(inside_ratio(p_exp, h) >= HELMET_INSIDE for h in helmets)

        # если каска есть -> SAFE (зелёный), иначе -> DANGER (красный)
        if has_helmet:
            safe += 1
            cv2.rectangle(img_bgr, (int(px1), int(py1)), (int(px2), int(py2)), (0, 255, 0), 3)
        else:
            danger += 1
            cv2.rectangle(img_bgr, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 3)

    return img_bgr, safe, danger


# -------------------- Init --------------------
try:
    model = load_model()
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()


# -------------------- Sidebar --------------------
st.sidebar.header("Настройки")
conf = st.sidebar.slider("Порог каски (helmet)", 0.05, 1.0, 0.25, 0.05)
person_conf = st.sidebar.slider("Порог человека (person)", 0.01, 1.0, 0.12, 0.01)
cfg_set(conf=conf, person_conf=person_conf)

st.sidebar.write("---")
st.sidebar.write("Классы модели:")
st.sidebar.code("\n".join([f"{i}: {n}" for i, n in model.names.items()]))


# -------------------- LIVE --------------------
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        cfg = cfg_get()
        out, _, _ = process_frame_helmet_only(img, model, cfg["conf"], cfg["person_conf"])
        return av.VideoFrame.from_ndarray(out, format="bgr24")


tab1, tab2 = st.tabs(["🎥 LIVE ВИДЕО", "📁 АНАЛИЗ ФОТО"])

with tab1:
    st.write("SAFE/DANGER считаются по каске: есть каска = зелёный, нет = красный.")
    webrtc_streamer(
        key="ppe-live",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": 12}, "audio": False},
        async_processing=True,
    )

with tab2:
    up_img = st.file_uploader("Загрузите фото", type=["jpg", "jpeg", "png"])
    if up_img:
        img_bgr = cv2.cvtColor(np.array(Image.open(up_img).convert("RGB")), cv2.COLOR_RGB2BGR)
        out, safe, danger = process_frame_helmet_only(img_bgr, model, conf, person_conf)
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.write(f"### SAFE (в каске): {safe} | DANGER (без каски): {danger}")
