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
st.title("🛡️ SafeGuard PRO — контроль каски")


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


# -------------------- thread-safe config for LIVE --------------------
_lock = threading.Lock()
CFG = {
    "conf_person": 0.12,
    "conf_helmet": 0.25,
    # matching params (можно не трогать)
    "x_margin": 0.15,      # запас по ширине person (15%)
    "y_top": 1.30,         # насколько высоко вверх от py1 искать каску (в долях высоты person)
    "y_bottom": 0.45,      # насколько вниз от py1 разрешаем каску (верхняя часть тела)
}


def cfg_set(**kwargs):
    with _lock:
        CFG.update(kwargs)


def cfg_get():
    with _lock:
        return dict(CFG)


# -------------------- helpers --------------------
def indicator_html(ok: bool, text: str):
    color = "#22c55e" if ok else "#ef4444"
    return f"""
    <div style="display:flex; align-items:center; gap:10px; padding:10px 12px; border-radius:12px;
                background: rgba(0,0,0,0.04); border:1px solid rgba(0,0,0,0.08);">
      <div style="width:16px; height:16px; border-radius:50%; background:{color};"></div>
      <div style="font-size:16px; font-weight:700;">{text}</div>
    </div>
    """


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


def box_center(b):
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def match_helmet_to_person(person_box, helmet_box, x_margin, y_top, y_bottom) -> bool:
    """
    person_box: xyxy
    helmet_box: xyxy
    Условие: каска должна быть по X в пределах person (с запасом)
             и по Y в "зоне головы" относительно верхней границы person.
    """
    px1, py1, px2, py2 = person_box
    pw = max(1.0, px2 - px1)
    ph = max(1.0, py2 - py1)

    hcx, hcy = box_center(helmet_box)

    # расширяем по X
    x1 = px1 - x_margin * pw
    x2 = px2 + x_margin * pw

    # зона по Y: от (py1 - y_top*ph) до (py1 + y_bottom*ph)
    y1 = py1 - y_top * ph
    y2 = py1 + y_bottom * ph

    return (x1 <= hcx <= x2) and (y1 <= hcy <= y2)


def process_frame(img_bgr, model, classes, conf_person, conf_helmet, x_margin, y_top, y_bottom):
    # общий conf минимальный, чтобы не срезать person
    pred_conf = min(conf_person, conf_helmet, 0.10)

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

    # рисуем каски зелёным
    for h in helmets:
        x1, y1, x2, y2 = map(int, h)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # назначаем каски людям: каска -> лучший (ближайший по X) person среди подходящих
    has_helmet = [False] * len(persons)

    for h in helmets:
        hcx, hcy = box_center(h)

        best_i = None
        best_dx = 1e18

        for i, p in enumerate(persons):
            if match_helmet_to_person(p, h, x_margin=x_margin, y_top=y_top, y_bottom=y_bottom):
                pcx, pcy = box_center(p)
                dx = abs(hcx - pcx)
                if dx < best_dx:
                    best_dx = dx
                    best_i = i

        if best_i is not None:
            has_helmet[best_i] = True

    safe = 0
    danger = 0

    # рисуем людей зелёный/красный
    for i, p in enumerate(persons):
        x1, y1, x2, y2 = map(int, p)
        if has_helmet[i]:
            safe += 1
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
        else:
            danger += 1
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # общий статус (по кадру)
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


# -------------------- init --------------------
try:
    classes = load_classes()
    model = load_model()
except Exception as e:
    st.error(f"Ошибка загрузки: {e}")
    st.stop()


# -------------------- sidebar --------------------
st.sidebar.header("Настройки")
conf_person = st.sidebar.slider("Порог человека (person)", 0.01, 1.0, 0.12, 0.01)
conf_helmet = st.sidebar.slider("Порог каски (helmet)", 0.05, 1.0, 0.25, 0.05)

# matching (можно не трогать)
x_margin = st.sidebar.slider("Запас по ширине (X)", 0.0, 0.6, 0.15, 0.05)
y_top = st.sidebar.slider("Зона вверх (Y)", 0.3, 2.5, 1.30, 0.10)
y_bottom = st.sidebar.slider("Зона вниз (Y)", 0.1, 1.2, 0.45, 0.05)

cfg_set(conf_person=conf_person, conf_helmet=conf_helmet, x_margin=x_margin, y_top=y_top, y_bottom=y_bottom)

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
            img,
            model,
            classes,
            cfg["conf_person"],
            cfg["conf_helmet"],
            cfg["x_margin"],
            cfg["y_top"],
            cfg["y_bottom"],
        )

        with self._l:
            self.last = {"safe": safe, "danger": danger, "ok": ok, "text": text}

        return av.VideoFrame.from_ndarray(out, format="bgr24")


# -------------------- tabs --------------------
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
            img_bgr, model, classes, conf_person, conf_helmet, x_margin, y_top, y_bottom
        )
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.markdown(indicator_html(ok, text), unsafe_allow_html=True)
        st.write(f"SAFE (в каске): {safe} | DANGER (без каски): {danger}")
