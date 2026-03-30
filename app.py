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


# ---------- Streamlit first ----------
st.set_page_config(page_title="SafeGuard PRO", layout="centered")
st.title("🛡️ SafeGuard PRO — контроль каски")


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "best.onnx"

IMGSZ = 640

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


# ---------- shared config for live thread ----------
_lock = threading.Lock()
CFG = {
    "conf": 0.15,          # минимальный conf для model.predict (чтобы ничего не отрезалось)
    "person_conf": 0.12,   # порог для person
    "helmet_conf": 0.25,   # порог для helmet
    "person_id": 0,
    "helmet_id": 1,
    # зона головы относительно person-box
    "x_margin": 0.20,
    "y_top": 1.80,
    "y_bottom": 0.55,
    "debug": False,
}


def cfg_set(**kwargs):
    with _lock:
        CFG.update(kwargs)


def cfg_get():
    with _lock:
        return dict(CFG)


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
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"best.onnx not found: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH), task="detect")


def get_names(model):
    # model.names обычно dict {id: name}
    names = getattr(model, "names", None)
    if isinstance(names, dict) and len(names) > 0:
        return names
    # fallback
    return {0: "class0"}


def head_zone(person_box, x_margin, y_top, y_bottom):
    px1, py1, px2, py2 = person_box
    pw = max(1.0, px2 - px1)
    ph = max(1.0, py2 - py1)
    x1 = px1 - x_margin * pw
    x2 = px2 + x_margin * pw
    y1 = py1 - y_top * ph
    y2 = py1 + y_bottom * ph
    return [x1, y1, x2, y2]


def center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_in(box, x, y):
    x1, y1, x2, y2 = box
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def process_frame(img_bgr, model, cfg):
    res = model.predict(
        img_bgr,
        conf=cfg["conf"],
        imgsz=IMGSZ,
        iou=0.3,
        verbose=False,
    )[0]

    persons = []
    helmets = []

    # debug: рисовать cls_id на всех боксах
    for b in res.boxes:
        cls_id = int(b.cls[0])
        score = float(b.conf[0])
        xyxy = b.xyxy[0].tolist()

        if cfg["debug"]:
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 255, 0), 1)
            cv2.putText(img_bgr, f"id{cls_id}:{score:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if cls_id == cfg["person_id"] and score >= cfg["person_conf"]:
            persons.append(xyxy)
        elif cls_id == cfg["helmet_id"] and score >= cfg["helmet_conf"]:
            helmets.append(xyxy)

    # рисуем каски зелёным
    for h in helmets:
        x1, y1, x2, y2 = map(int, h)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    safe = 0
    danger = 0

    # назначаем каски людям по зоне головы
    for p in persons:
        px1, py1, px2, py2 = p
        hz = head_zone(p, cfg["x_margin"], cfg["y_top"], cfg["y_bottom"])

        has_helmet = False
        for h in helmets:
            hx, hy = center(h)
            if point_in(hz, hx, hy):
                has_helmet = True
                break

        if has_helmet:
            safe += 1
            cv2.rectangle(img_bgr, (int(px1), int(py1)), (int(px2), int(py2)), (0, 255, 0), 3)
        else:
            danger += 1
            cv2.rectangle(img_bgr, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 3)

    if len(persons) == 0:
        ok = False
        text = "ЧЕЛОВЕК НЕ ОБНАРУЖЕН"
    elif danger == 0 and safe > 0:
        ok = True
        text = "ЕСТЬ КАСКА"
    else:
        ok = False
        text = "НЕТ КАСКИ"

    return img_bgr, safe, danger, ok, text


# ---------- init ----------
try:
    model = load_model()
    names = get_names(model)
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()


# ---------- sidebar ----------
st.sidebar.header("Настройки")
st.sidebar.write("Выбери правильные class_id (это решает твою проблему).")

ids = sorted(list(names.keys()))

def fmt(i):
    return f"{i}: {names.get(i,'')}"


person_id = st.sidebar.selectbox("PERSON class_id", ids, index=0, format_func=fmt)
helmet_id = st.sidebar.selectbox("HELMET class_id", ids, index=min(1, len(ids)-1), format_func=fmt)

person_conf = st.sidebar.slider("Порог person", 0.01, 1.0, 0.12, 0.01)
helmet_conf = st.sidebar.slider("Порог helmet", 0.01, 1.0, 0.25, 0.01)

x_margin = st.sidebar.slider("X запас (голова)", 0.0, 0.6, 0.20, 0.05)
y_top = st.sidebar.slider("Y вверх (голова)", 0.3, 2.8, 1.80, 0.10)
y_bottom = st.sidebar.slider("Y вниз (голова)", 0.1, 1.2, 0.55, 0.05)

debug = st.sidebar.checkbox("DEBUG: показать id на всех боксах", value=False)

cfg_set(
    person_id=person_id,
    helmet_id=helmet_id,
    person_conf=person_conf,
    helmet_conf=helmet_conf,
    x_margin=x_margin,
    y_top=y_top,
    y_bottom=y_bottom,
    debug=debug,
)

st.sidebar.write("---")
st.sidebar.write("names из модели:")
st.sidebar.code("\n".join([f"{i}: {names[i]}" for i in ids]))


# ---------- webrtc ----------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last = {"safe": 0, "danger": 0, "ok": False, "text": "—"}
        self._l = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        cfg = cfg_get()
        out, safe, danger, ok, text = process_frame(img, model, cfg)
        with self._l:
            self.last = {"safe": safe, "danger": danger, "ok": ok, "text": text}
        return av.VideoFrame.from_ndarray(out, format="bgr24")


tab1, tab2 = st.tabs(["🎥 LIVE", "📁 Фото"])

with tab1:
    ctx = webrtc_streamer(
        key="ppe-live",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
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
        img = cv2.cvtColor(np.array(Image.open(up_img).convert("RGB")), cv2.COLOR_RGB2BGR)
        cfg = cfg_get()
        out, safe, danger, ok, text = process_frame(img, model, cfg)
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.markdown(indicator_html(ok, text), unsafe_allow_html=True)
        st.write(f"SAFE: {safe} | DANGER: {danger}")
