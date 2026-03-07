import os
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
st.title("🛡️ SafeGuard PRO — СИЗ (зелёный/красный)")

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "best.onnx"
CLASSES_PATH = ROOT / "classes.txt"

IMGSZ = 640


# -------------------- TURN/ICE (чтобы LIVE работал на Streamlit Cloud) --------------------
# Если добавишь свои TURN в Secrets — они будут использованы.
def build_rtc_configuration():
    # дефолтный публичный TURN (может быть медленным, но часто помогает)
    ice = [
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

    # опционально: свои TURN из Streamlit Secrets
    # В Streamlit Cloud -> App settings -> Secrets:
    # TURN_URLS='["turn:turn.example.com:3478?transport=udp","turns:turn.example.com:5349?transport=tcp"]'
    # TURN_USERNAME="user"
    # TURN_CREDENTIAL="pass"
    try:
        if "TURN_URLS" in st.secrets and "TURN_USERNAME" in st.secrets and "TURN_CREDENTIAL" in st.secrets:
            import json
            ice = [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {
                    "urls": json.loads(st.secrets["TURN_URLS"]),
                    "username": st.secrets["TURN_USERNAME"],
                    "credential": st.secrets["TURN_CREDENTIAL"],
                },
            ]
    except Exception:
        pass

    return RTCConfiguration({"iceServers": ice})


RTC_CONFIGURATION = build_rtc_configuration()


# -------------------- Shared config for video thread --------------------
_lock = threading.Lock()
CFG = {"conf_ppe": 0.30, "conf_person": 0.15, "strict": False}


def cfg_set(**kwargs):
    with _lock:
        CFG.update(kwargs)


def cfg_get():
    with _lock:
        return dict(CFG)


# -------------------- Model --------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"best.onnx not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH), task="detect")

    # если в ONNX вдруг нет names, подхватим из classes.txt
    if (not hasattr(model, "names")) or (not model.names):
        if CLASSES_PATH.exists():
            names = [x.strip() for x in CLASSES_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
            model.names = {i: n for i, n in enumerate(names)}

    return model


def norm_label(s: str) -> str:
    return (s or "").strip().lower().replace("_", "-").replace(" ", "")


def is_person(label: str) -> bool:
    l = norm_label(label)
    return l == "person" or "person" in l or "human" in l


def is_no_helmet(label: str) -> bool:
    return "no-helmet" in norm_label(label)


def is_no_vest(label: str) -> bool:
    return "no-vest" in norm_label(label)


def is_helmet(label: str) -> bool:
    # ВАЖНО: no-helmet содержит "helmet"
    if is_no_helmet(label):
        return False
    l = norm_label(label)
    return "helmet" in l or "hardhat" in l or "hard-hat" in l


def is_vest(label: str) -> bool:
    # ВАЖНО: no-vest содержит "vest"
    if is_no_vest(label):
        return False
    l = norm_label(label)
    return "vest" in l or "jacket" in l


def intersects(a, b) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (bx2 < ax1 or bx1 > ax2 or by2 < ay1 or by1 > ay2)


def center_inside(person_box, obj_box) -> bool:
    px1, py1, px2, py2 = person_box
    ox1, oy1, ox2, oy2 = obj_box
    cx, cy = (ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0
    return (px1 < cx < px2) and (py1 < cy < py2)


def expand_person_up(p, img_h, ratio=0.55):
    """person bbox часто обрезает голову -> расширяем вверх для каски"""
    x1, y1, x2, y2 = p
    h = max(1.0, y2 - y1)
    y1 = max(0.0, y1 - ratio * h)
    y2 = min(float(img_h - 1), y2)
    return [x1, y1, x2, y2]


def process_frame_logic(img_bgr, model, conf_ppe, conf_person, strict):
    # общий conf делаем минимальным, чтобы person не отрезало
    predict_conf = min(conf_ppe, conf_person, 0.10)

    results = model.predict(img_bgr, conf=predict_conf, imgsz=IMGSZ, iou=0.3, verbose=False)
    boxes = results[0].boxes

    people = []
    helmets = []
    vests = []
    direct_viol = []  # no-helmet/no-vest

    for box in boxes:
        cls_id = int(box.cls[0])
        label = model.names.get(cls_id, str(cls_id))
        score = float(box.conf[0])
        coords = box.xyxy[0].tolist()

        if is_person(label):
            if score >= conf_person:
                people.append(coords)
        elif is_no_helmet(label) or is_no_vest(label):
            if score >= conf_ppe:
                direct_viol.append(coords)
        elif is_helmet(label):
            if score >= conf_ppe:
                helmets.append(coords)
        elif is_vest(label):
            if score >= conf_ppe:
                vests.append(coords)

    safe = 0
    danger = 0
    img_h = img_bgr.shape[0]

    # рисуем ТОЛЬКО людей зелёный/красный
    for p in people:
        px1, py1, px2, py2 = p
        p_exp = expand_person_up(p, img_h)

        # прямые нарушения
        if any(intersects(p_exp, v) for v in direct_viol):
            danger += 1
            cv2.rectangle(img_bgr, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 3)
            continue

        has_helmet = any(center_inside(p_exp, h) for h in helmets)
        has_vest = any(center_inside(p, v) for v in vests)

        ok = (has_helmet and has_vest) if strict else (has_helmet or has_vest)

        if ok:
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
conf_ppe = st.sidebar.slider("Порог СИЗ (helmet/vest)", 0.05, 1.0, 0.30, 0.05)
conf_person = st.sidebar.slider("Порог PERSON", 0.01, 1.0, 0.15, 0.01)
strict = st.sidebar.checkbox("SAFE только если каска+жилет", value=False)
cfg_set(conf_ppe=conf_ppe, conf_person=conf_person, strict=strict)

st.sidebar.write("---")
st.sidebar.write("Classes:")
try:
    st.sidebar.code("\n".join([f"{i}: {n}" for i, n in model.names.items()]))
except Exception:
    st.sidebar.write("no names")


# -------------------- LIVE --------------------
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        cfg = cfg_get()
        out, _, _ = process_frame_logic(img, model, cfg["conf_ppe"], cfg["conf_person"], cfg["strict"])
        return av.VideoFrame.from_ndarray(out, format="bgr24")


tab1, tab2 = st.tabs(["🎥 LIVE ВИДЕО", "📁 АНАЛИЗ ФОТО"])

with tab1:
    webrtc_streamer(
        key="ppe-live",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with tab2:
    up_img = st.file_uploader("Загрузите фото", type=["jpg", "jpeg", "png"])
    if up_img:
        img_bgr = cv2.cvtColor(np.array(Image.open(up_img).convert("RGB")), cv2.COLOR_RGB2BGR)
        res, safe, danger = process_frame_logic(img_bgr, model, conf_ppe, conf_person, strict)
        st.image(cv2.cvtColor(res, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.write(f"### SAFE: {safe} | DANGER: {danger}")
