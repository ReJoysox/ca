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


# -------------------- Streamlit (первый st.*) --------------------
st.set_page_config(page_title="SafeGuard PRO", layout="centered")
st.title("🛡️ SafeGuard PRO — контроль каски (зелёный/красный)")


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "best.onnx"
CLASSES_PATH = ROOT / "classes.txt"
IMGSZ = 640


# -------------------- WebRTC TURN (для Streamlit Cloud) --------------------
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


# -------------------- Thread-safe config for LIVE --------------------
_lock = threading.Lock()
CFG = {
    "conf_helmet": 0.25,
    "conf_person": 0.12,
    "expand_up": 0.80,     # сильнее расширяем person вверх
    "min_inside": 0.01,    # насколько мало пересечение каски с человеком допускаем
}


def cfg_set(**kwargs):
    with _lock:
        CFG.update(kwargs)


def cfg_get():
    with _lock:
        return dict(CFG)


# -------------------- Loaders --------------------
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


# -------------------- Helpers --------------------
def label_from_id(cls_id: int, classes: list[str]) -> str:
    if 0 <= cls_id < len(classes):
        return classes[cls_id]
    return f"class{cls_id}"


def expand_person_up(p, img_h, ratio):
    x1, y1, x2, y2 = p
    h = max(1.0, y2 - y1)
    y1 = max(0.0, y1 - ratio * h)
    y2 = min(float(img_h - 1), y2)
    return [x1, y1, x2, y2]


def box_area(b) -> float:
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersect_area(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def inside_ratio(person_box, obj_box) -> float:
    # доля площади obj (каски) внутри person
    inter = intersect_area(person_box, obj_box)
    return inter / (box_area(obj_box) + 1e-9)


def helmet_center_in_person(person_box, helmet_box) -> bool:
    px1, py1, px2, py2 = person_box
    hx1, hy1, hx2, hy2 = helmet_box
    cx, cy = (hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0
    return (px1 <= cx <= px2) and (py1 <= cy <= py2)


def indicator_html(ok: bool, text: str):
    color = "#22c55e" if ok else "#ef4444"
    return f"""
    <div style="display:flex; align-items:center; gap:10px; padding:10px 12px; border-radius:12px;
                background: rgba(0,0,0,0.04); border:1px solid rgba(0,0,0,0.08);">
      <div style="width:16px; height:16px; border-radius:50%; background:{color};"></div>
      <div style="font-size:16px; font-weight:700;">{text}</div>
    </div>
    """


# -------------------- Core logic --------------------
def process_frame(img_bgr, model, classes, conf_helmet, conf_person, expand_up, min_inside):
    # общий conf минимальный, чтобы person не срезался
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

    img_h = img_bgr.shape[0]

    # рисуем каски зелёным
    for h in helmets:
        x1, y1, x2, y2 = map(int, h)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # назначаем каски людям (чтобы не “терялись”)
    has_helmet_for_person = [False] * len(persons)

    expanded_people = [expand_person_up(p, img_h, expand_up) for p in persons]

    for h in helmets:
        best_i = -1
        best_score = 0.0
        for i, pexp in enumerate(expanded_people):
            score = inside_ratio(pexp, h)
            if helmet_center_in_person(pexp, h):
                score = max(score, 1.0)  # если центр внутри — это почти точно тот человек
            if score > best_score:
                best_score = score
                best_i = i

        if best_i != -1 and best_score >= min_inside:
            has_helmet_for_person[best_i] = True

    safe = 0
    danger = 0

    # рисуем людей зелёный/красный
    for i, p in enumerate(persons):
        x1, y1, x2, y2 = map(int, p)
        if has_helmet_for_person[i]:
            safe += 1
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
        else:
            danger += 1
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # общий статус
    if len(persons) == 0:
        ok = False
        text = "ЧЕЛОВЕК НЕ ОБНАРУЖЕН"
    elif danger == 0 and safe > 0:
        ok = True
        text = "ЕСТЬ КАСКА"
    elif safe == 0 and danger > 0:
        ok = False
        text = "НЕТ КАСКИ"
    else:
        ok = False
        text = "ЧАСТИЧНО: ЕСТЬ/НЕТ"

    return img_bgr, safe, danger, ok, text


# -------------------- Init --------------------
try:
    classes = load_classes()
    model = load_model()
except Exception as e:
    st.error(f"Ошибка загрузки: {e}")
    st.stop()


# -------------------- Sidebar --------------------
st.sidebar.header("Настройки")
conf_helmet = st.sidebar.slider("Порог каски", 0.05, 1.0, 0.25, 0.05)
conf_person = st.sidebar.slider("Порог человека", 0.01, 1.0, 0.12, 0.01)
expand_up = st.sidebar.slider("Расширение person вверх", 0.20, 1.50, 0.80, 0.05)
min_inside = st.sidebar.slider("Привязка каски к человеку", 0.001, 0.10, 0.01, 0.001)

cfg_set(conf_helmet=conf_helmet, conf_person=conf_person, expand_up=expand_up, min_inside=min_inside)

st.sidebar.write("---")
st.sidebar.write("classes.txt:")
st.sidebar.code("\n".join([f"{i}: {n}" for i, n in enumerate(classes)]))


# -------------------- LIVE processor --------------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last = {"safe": 0, "danger": 0, "ok": False, "text": "—"}
        self._l = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        cfg = cfg_get()

        out, safe, danger, ok, text = process_frame(
            img, model, classes,
            cfg["conf_helmet"], cfg["conf_person"],
            cfg["expand_up"], cfg["min_inside"]
        )

        with self._l:
            self.last = {"safe": safe, "danger": danger, "ok": ok, "text": text}

        return av.VideoFrame.from_ndarray(out, format="bgr24")


# -------------------- Tabs --------------------
tab1, tab2 = st.tabs(["🎥 LIVE ВИДЕО", "📁 АНАЛИЗ ФОТО"])

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
        st.write(f"SAFE (в каске): {info['safe']} | DANGER (без каски): {info['danger']}")
    else:
        st.markdown(indicator_html(False, "LIVE не запущен"), unsafe_allow_html=True)

with tab2:
    up_img = st.file_uploader("Загрузите фото", type=["jpg", "jpeg", "png"])
    if up_img:
        img_bgr = cv2.cvtColor(np.array(Image.open(up_img).convert("RGB")), cv2.COLOR_RGB2BGR)
        out, safe, danger, ok, text = process_frame(
            img_bgr, model, classes, conf_helmet, conf_person, expand_up, min_inside
        )
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.markdown(indicator_html(ok, text), unsafe_allow_html=True)
        st.write(f"SAFE (в каске): {safe} | DANGER (без каски): {danger}")
