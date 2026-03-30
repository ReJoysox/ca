import os
os.environ["ORT_SKIP_OPSET_VALIDATION"] = "1"
os.environ["ORT_LOGGING_LEVEL"] = "3"

from pathlib import Path
import threading

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration


# ================= Streamlit (первый st.*) =================
st.set_page_config(page_title="SafeGuard PRO", layout="centered")

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "best.onnx"
CLASSES_PATH = ROOT / "classes.txt"
IMGSZ = 640


# ================= TURN (для Streamlit Cloud) =================
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


# ================= Thread-safe config =================
_lock = threading.Lock()
CFG = {
    "conf_helmet": 0.25,
    "conf_person": 0.12,
    "helmet_inside_ratio": 0.03,
    "person_expand_up": 0.65,
}


def cfg_set(**kwargs):
    with _lock:
        CFG.update(kwargs)


def cfg_get():
    with _lock:
        return dict(CFG)


# ================= UI helpers =================
def indicator_html(ok: bool, text: str):
    color = "#22c55e" if ok else "#ef4444"
    return f"""
    <div style="display:flex; align-items:center; gap:10px; padding:10px 12px; border-radius:12px;
                background: rgba(0,0,0,0.04); border:1px solid rgba(0,0,0,0.08);">
      <div style="width:16px; height:16px; border-radius:50%; background:{color};"></div>
      <div style="font-size:16px; font-weight:700;">{text}</div>
    </div>
    """


def expand_person_up(p, img_h, ratio):
    x1, y1, x2, y2 = p
    h = max(1.0, y2 - y1)
    y1 = max(0.0, y1 - ratio * h)
    y2 = min(float(img_h - 1), y2)
    return [x1, y1, x2, y2]


def intersect_area(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_area(b) -> float:
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def inside_ratio(person_box, obj_box) -> float:
    inter = intersect_area(person_box, obj_box)
    return inter / (box_area(obj_box) + 1e-9)


# ================= Load classes/model =================
@st.cache_resource
def load_classes():
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(f"classes.txt not found: {CLASSES_PATH}")
    return [x.strip() for x in CLASSES_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"best.onnx does not exist: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH), task="detect")


def label_from_id(cls_id: int, classes: list[str]) -> str:
    if 0 <= cls_id < len(classes):
        return classes[cls_id]
    return f"class{cls_id}"


# ================= Core logic =================
def process_frame(img_bgr, model, classes, conf_helmet, conf_person, helmet_inside_ratio, person_expand_up):
    pred_conf = min(conf_helmet, conf_person, 0.10)

    res = model.predict(img_bgr, conf=pred_conf, imgsz=IMGSZ, iou=0.3, verbose=False)[0]
    boxes = res.boxes

    persons = []
    helmets = []

    for b in boxes:
        cls_id = int(b.cls[0])
        label = label_from_id(cls_id, classes)
        score = float(b.conf[0])
        xyxy = b.xyxy[0].tolist()

        if label == "person" and score >= conf_person:
            persons.append(xyxy)
        elif label == "helmet" and score >= conf_helmet:
            helmets.append(xyxy)

    safe = 0
    danger = 0
    img_h = img_bgr.shape[0]

    # каски рисуем зелёным
    for h in helmets:
        x1, y1, x2, y2 = map(int, h)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for p in persons:
        px1, py1, px2, py2 = p
        p_exp = expand_person_up(p, img_h, person_expand_up)

        has_helmet = any(inside_ratio(p_exp, h) >= helmet_inside_ratio for h in helmets)

        if has_helmet:
            safe += 1
            cv2.rectangle(img_bgr, (int(px1), int(py1)), (int(px2), int(py2)), (0, 255, 0), 3)
        else:
            danger += 1
            cv2.rectangle(img_bgr, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 3)

    if len(persons) == 0:
        status_ok = False
        status_text = "ЧЕЛОВЕК НЕ ОБНАРУЖЕН"
    else:
        status_ok = (danger == 0 and safe > 0)
        status_text = "ЕСТЬ СИЗ" if status_ok else "НЕТ СИЗ"

    return img_bgr, safe, danger, status_ok, status_text


# ================= UI =================
st.title("🛡️ SafeGuard PRO")

try:
    classes = load_classes()
    model = load_model()
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

st.sidebar.header("Настройки")
conf_helmet = st.sidebar.slider("Порог каски (helmet)", 0.05, 1.0, 0.25, 0.05)
conf_person = st.sidebar.slider("Порог человека (person)", 0.01, 1.0, 0.12, 0.01)
helmet_inside_ratio = st.sidebar.slider("Привязка каски к человеку", 0.01, 0.30, 0.03, 0.01)
person_expand_up = st.sidebar.slider("Расширение person вверх", 0.10, 1.20, 0.65, 0.05)

cfg_set(
    conf_helmet=conf_helmet,
    conf_person=conf_person,
    helmet_inside_ratio=helmet_inside_ratio,
    person_expand_up=person_expand_up,
)

st.sidebar.write("---")
st.sidebar.write("classes.txt:")
st.sidebar.code("\n".join([f"{i}: {n}" for i, n in enumerate(classes)]))

tab1, tab2 = st.tabs(["🎥 LIVE ВИДЕО", "📁 АНАЛИЗ ФОТО"])


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last = {"safe": 0, "danger": 0, "ok": False, "text": "—"}
        self._l = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        cfg = cfg_get()

        out, safe, danger, ok, text = process_frame(
            img,
            model,
            classes,
            cfg["conf_helmet"],
            cfg["conf_person"],
            cfg["helmet_inside_ratio"],
            cfg["person_expand_up"],
        )

        with self._l:
            self.last = {"safe": safe, "danger": danger, "ok": ok, "text": text}

        return av.VideoFrame.from_ndarray(out, format="bgr24")


with tab1:
    ctx = webrtc_streamer(
        key="ppe-live",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": 12}, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        with ctx.video_processor._l:
            info = dict(ctx.video_processor.last)
        st.markdown(indicator_html(info["ok"], info["text"]), unsafe_allow_html=True)
        st.write(f"SAFE: {info['safe']} | DANGER: {info['danger']}")
    else:
        st.markdown(indicator_html(False, "LIVE не запущен"), unsafe_allow_html=True)


with tab2:
    up_img = st.file_uploader("Загрузите фото", type=["jpg", "jpeg", "png"])
    if up_img:
        img_bgr = cv2.cvtColor(np.array(Image.open(up_img).convert("RGB")), cv2.COLOR_RGB2BGR)
        out, safe, danger, ok, text = process_frame(
            img_bgr, model, classes, conf_helmet, conf_person, helmet_inside_ratio, person_expand_up
        )
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.markdown(indicator_html(ok, text), unsafe_allow_html=True)
        st.write(f"SAFE: {safe} | DANGER: {danger}")
