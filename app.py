from pathlib import Path
import sys

BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))  # важно для .exe
MODEL_PATH = str(Path(__file__).parent / "best.onnx")
CLASSES_PATH = BASE_DIR / "classes.txt"
import os

# До onnxruntime/ultralytics
os.environ["ORT_SKIP_OPSET_VALIDATION"] = "1"
os.environ["ORT_LOGGING_LEVEL"] = "3"

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av


st.set_page_config(page_title="SafeGuard PRO", layout="centered")
st.title("🛡️ SafeGuard ИИ: Система мониторинга")

IMGSZ = 640  # твоя ONNX модель ожидает 640x640


@st.cache_resource
def load_model():
    return YOLO("best.onnx", task="detect")


def norm_label(s: str) -> str:
    return (s or "").strip().lower().replace("_", "-").replace(" ", "")


def is_person(label: str) -> bool:
    l = norm_label(label)
    return l == "person" or "person" in l or l == "human" or "human" in l


def is_no_helmet(label: str) -> bool:
    return "no-helmet" in norm_label(label)


def is_no_vest(label: str) -> bool:
    return "no-vest" in norm_label(label)


def is_helmet(label: str) -> bool:
    # важно: "no-helmet" содержит "helmet"
    if is_no_helmet(label):
        return False
    l = norm_label(label)
    return ("helmet" in l) or ("hardhat" in l) or ("hard-hat" in l)


def is_vest(label: str) -> bool:
    # важно: "no-vest" содержит "vest"
    if is_no_vest(label):
        return False
    l = norm_label(label)
    return ("vest" in l) or ("jacket" in l)


def center_inside(person_box, obj_box) -> bool:
    px1, py1, px2, py2 = person_box
    ox1, oy1, ox2, oy2 = obj_box
    cx, cy = (ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0
    return (px1 < cx < px2) and (py1 < cy < py2)


def expand_person_up(p, img_h, ratio=0.55):
    """Расширяем person вверх, потому что person-бокс часто 'обрезает' голову."""
    x1, y1, x2, y2 = p
    h = max(1.0, y2 - y1)
    y1 = max(0.0, y1 - ratio * h)
    y2 = min(float(img_h - 1), y2)
    return [x1, y1, x2, y2]


def intersects(a, b) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (bx2 < ax1 or bx1 > ax2 or by2 < ay1 or by1 > ay2)


def process_frame_logic(img_bgr, model, conf_ppe, conf_person, strict_both: bool):
    """
    strict_both=False -> SAFE если есть каска ИЛИ жилет
    strict_both=True  -> SAFE только если есть каска И жилет
    """
    # ВАЖНО: общий conf в predict ставим минимальный из порогов, чтобы не "срезать" person
    predict_conf = min(conf_ppe, conf_person)
    results = model.predict(img_bgr, conf=predict_conf, imgsz=IMGSZ, iou=0.3, verbose=False)
    boxes = results[0].boxes

    people = []
    helmets = []
    vests = []
    direct_viol = []  # no-helmet/no-vest (если модель их дает)

    for box in boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        coords = box.xyxy[0].tolist()
        score = float(box.conf[0])

        if is_person(label):
            if score >= conf_person:
                people.append(coords)
        elif is_no_helmet(label) or is_no_vest(label):
            if score >= conf_ppe:
                direct_viol.append({"label": label, "coords": coords, "score": score})
        elif is_helmet(label):
            if score >= conf_ppe:
                helmets.append(coords)
        elif is_vest(label):
            if score >= conf_ppe:
                vests.append(coords)

    safe_count = 0
    danger_count = 0
    img_h, img_w = img_bgr.shape[:2]

    # Рисуем ТОЛЬКО людей: зелёный/красный
    for p in people:
        px1, py1, px2, py2 = p
        p_exp = expand_person_up(p, img_h, ratio=0.55)

        # Если "прямое нарушение" пересекается с человеком — считаем нарушителем
        if any(intersects(p_exp, v["coords"]) for v in direct_viol):
            danger_count += 1
            cv2.rectangle(img_bgr, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 3)
            continue

        has_helmet = any(center_inside(p_exp, h) for h in helmets)
        has_vest = any(center_inside(p, v) for v in vests)

        ok = (has_helmet and has_vest) if strict_both else (has_helmet or has_vest)

        if ok:
            safe_count += 1
            cv2.rectangle(img_bgr, (int(px1), int(py1)), (int(px2), int(py2)), (0, 255, 0), 3)
        else:
            danger_count += 1
            cv2.rectangle(img_bgr, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 3)

    return img_bgr, safe_count, danger_count


# ---------------- Main ----------------
try:
    model = load_model()
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

st.sidebar.header("Настройки")
conf_ppe = st.sidebar.slider("Порог для СИЗ (helmet/vest)", 0.05, 1.0, 0.30, 0.05)
conf_person = st.sidebar.slider("Порог для человека (person)", 0.01, 1.0, 0.15, 0.01)
strict_mode = st.sidebar.checkbox("Строго: SAFE только если КАСКА + ЖИЛЕТ", value=False)

st.sidebar.write("---")
st.sidebar.write("Классы модели:")
st.sidebar.code("\n".join([f"{i}: {n}" for i, n in model.names.items()]))

tab1, tab2 = st.tabs(["🎥 LIVE ВИДЕО", "📁 АНАЛИЗ ФОТО"])


class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed, _, _ = process_frame_logic(img, model, conf_ppe, conf_person, strict_mode)
        return av.VideoFrame.from_ndarray(processed, format="bgr24")


with tab1:
    webrtc_streamer(
        key="ppe-safe",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": 12}, "audio": False},
        async_processing=True,
    )

with tab2:
    up_img = st.file_uploader("Загрузите фото", type=["jpg", "jpeg", "png"])
    if up_img:
        img_cv = cv2.cvtColor(np.array(Image.open(up_img).convert("RGB")), cv2.COLOR_RGB2BGR)
        res_cv, safe, bad = process_frame_logic(img_cv, model, conf_ppe, conf_person, strict_mode)

        st.image(cv2.cvtColor(res_cv, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.write(f"### SAFE: {safe} | DANGER: {bad}")