import os
# важно: до импорта onnxruntime
os.environ["ORT_SKIP_OPSET_VALIDATION"] = "1"
os.environ["ORT_LOGGING_LEVEL"] = "3"

from pathlib import Path
import threading

import av
import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration


# -------------------- Streamlit (должно быть первым st.*) --------------------
st.set_page_config(page_title="SafeGuard PRO", layout="centered")
st.title("SafeGuard PRO — контроль СИЗ")


# -------------------- Paths --------------------
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "best.onnx"
CLASSES_PATH = ROOT / "classes.txt"


# -------------------- WebRTC TURN (чтобы работало на Streamlit Cloud) --------------------
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
_cfg_lock = threading.Lock()
CFG = {
    "conf_ppe": 0.30,      # порог для каски/жилета
    "conf_person": 0.15,   # порог для person
    "strict": False,       # SAFE только если каска+жилет
    "draw_counts": False,  # рисовать счётчик на кадре (для LIVE)
}


def cfg_set(**kwargs):
    with _cfg_lock:
        CFG.update(kwargs)


def cfg_get():
    with _cfg_lock:
        return dict(CFG)


# -------------------- Classes & model --------------------
@st.cache_resource
def load_classes():
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(f"classes.txt not found: {CLASSES_PATH}")
    return [x.strip() for x in CLASSES_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]


@st.cache_resource
def load_session():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"best.onnx not found: {MODEL_PATH}")

    sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    input_name = inp.name

    imgsz = 640
    try:
        shape = inp.shape  # [1,3,H,W]
        h, w = shape[2], shape[3]
        if isinstance(h, int) and isinstance(w, int) and h == w:
            imgsz = int(h)
    except Exception:
        pass

    return sess, input_name, imgsz


# -------------------- Label helpers --------------------
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
    # КЛЮЧ: no-helmet содержит helmet -> исключаем
    if is_no_helmet(label):
        return False
    l = norm_label(label)
    return ("helmet" in l) or ("hardhat" in l) or ("hard-hat" in l)


def is_vest(label: str) -> bool:
    # КЛЮЧ: no-vest содержит vest -> исключаем
    if is_no_vest(label):
        return False
    l = norm_label(label)
    return ("vest" in l) or ("jacket" in l)


# -------------------- Geometry --------------------
def letterbox(img_bgr, new_shape, color=(114, 114, 114)):
    h, w = img_bgr.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_w = new_shape - nw
    pad_h = new_shape - nh
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, r, (left, top)


def expand_person_up(p, img_h, ratio=0.55):
    """Расширяем person вверх, потому что person-бокс часто не включает голову."""
    x1, y1, x2, y2 = p
    h = max(1.0, y2 - y1)
    y1 = max(0.0, y1 - ratio * h)
    y2 = min(float(img_h - 1), y2)
    return [x1, y1, x2, y2]


def center_inside(person_box, obj_box) -> bool:
    px1, py1, px2, py2 = person_box
    ox1, oy1, ox2, oy2 = obj_box
    cx, cy = (ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0
    return (px1 < cx < px2) and (py1 < cy < py2)


def intersects(a, b) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (bx2 < ax1 or bx1 > ax2 or by2 < ay1 or by1 > ay2)


# -------------------- ONNX inference --------------------
def infer(img_bgr, sess, input_name, imgsz):
    h0, w0 = img_bgr.shape[:2]
    img_lb, r, (padx, pady) = letterbox(img_bgr, imgsz)
    x = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]  # 1x3xHxW
    out = sess.run(None, {input_name: x})[0]
    return out, (h0, w0), r, (padx, pady), imgsz


def parse_dets_nx6(out, class_names, conf_min, orig_hw, r, pad, imgsz):
    """
    Ожидаем выход (N,6) или (1,N,6):
    x1,y1,x2,y2,score,cls
    """
    h0, w0 = orig_hw
    padx, pady = pad
    pred = np.asarray(out)

    if pred.ndim == 3 and pred.shape[-1] == 6:
        pred = pred[0]

    if not (pred.ndim == 2 and pred.shape[1] == 6):
        return None  # неизвестный формат

    boxes = pred[:, :4].astype(np.float32)
    scores = pred[:, 4].astype(np.float32)
    cls_id = pred[:, 5].astype(np.int64)

    keep = scores >= conf_min
    boxes, scores, cls_id = boxes[keep], scores[keep], cls_id[keep]

    dets = []
    for b, sc, ci in zip(boxes, scores, cls_id):
        x1, y1, x2, y2 = b.tolist()

        # если вдруг нормализовано 0..1
        if max(x2, y2) <= 1.5:
            x1, x2 = x1 * imgsz, x2 * imgsz
            y1, y2 = y1 * imgsz, y2 * imgsz

        # letterbox -> original
        x1 = (x1 - padx) / r
        y1 = (y1 - pady) / r
        x2 = (x2 - padx) / r
        y2 = (y2 - pady) / r

        x1 = float(np.clip(x1, 0, w0 - 1))
        y1 = float(np.clip(y1, 0, h0 - 1))
        x2 = float(np.clip(x2, 0, w0 - 1))
        y2 = float(np.clip(y2, 0, h0 - 1))

        label = class_names[int(ci)] if int(ci) < len(class_names) else f"class{int(ci)}"
        dets.append({"label": label, "coords": [x1, y1, x2, y2], "score": float(sc)})

    return dets


def detect(img_bgr, sess, input_name, imgsz, class_names, conf_min):
    out, orig_hw, r, pad, imgsz = infer(img_bgr, sess, input_name, imgsz)

    dets = parse_dets_nx6(out, class_names, conf_min=conf_min, orig_hw=orig_hw, r=r, pad=pad, imgsz=imgsz)
    if dets is None:
        raise RuntimeError("Неизвестный формат выхода ONNX. Нужен выход Nx6 (x1,y1,x2,y2,score,cls).")

    return dets


# -------------------- Main logic (зелёный/красный) --------------------
def process_frame(img_bgr, sess, input_name, imgsz, class_names, conf_ppe, conf_person, strict, draw_counts):
    conf_min = min(conf_ppe, conf_person, 0.10)
    dets = detect(img_bgr, sess, input_name, imgsz, class_names, conf_min=conf_min)

    people = [d["coords"] for d in dets if is_person(d["label"]) and d["score"] >= conf_person]
    helmets = [d["coords"] for d in dets if is_helmet(d["label"]) and d["score"] >= conf_ppe]
    vests = [d["coords"] for d in dets if is_vest(d["label"]) and d["score"] >= conf_ppe]
    no_boxes = [d["coords"] for d in dets if (is_no_helmet(d["label"]) or is_no_vest(d["label"])) and d["score"] >= conf_ppe]

    safe = 0
    danger = 0
    img_h = img_bgr.shape[0]

    for p in people:
        px1, py1, px2, py2 = p
        p_exp = expand_person_up(p, img_h, ratio=0.55)

        # если пересёкся no-helmet/no-vest -> нарушитель
        if any(intersects(p_exp, nb) for nb in no_boxes):
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

    if draw_counts:
        cv2.rectangle(img_bgr, (0, 0), (160, 46), (0, 0, 0), -1)
        cv2.putText(img_bgr, f"SAFE:{safe}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img_bgr, f"DANG:{danger}", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return img_bgr, safe, danger


# -------------------- Init --------------------
try:
    class_names = load_classes()
    sess, input_name, imgsz = load_session()
except Exception as e:
    st.error(f"Ошибка инициализации: {e}")
    st.stop()


# -------------------- Sidebar --------------------
st.sidebar.header("Настройки")
conf_ppe = st.sidebar.slider("Порог СИЗ (helmet/vest)", 0.05, 1.0, 0.30, 0.05)
conf_person = st.sidebar.slider("Порог PERSON", 0.01, 1.0, 0.15, 0.01)
strict = st.sidebar.checkbox("SAFE только если каска + жилет", value=False)
draw_counts = st.sidebar.checkbox("Показывать счётчик на LIVE", value=False)

cfg_set(conf_ppe=conf_ppe, conf_person=conf_person, strict=strict, draw_counts=draw_counts)

st.sidebar.write("---")
st.sidebar.write(f"Input: {imgsz}x{imgsz}")
st.sidebar.write("Classes:")
st.sidebar.code("\n".join([f"{i}: {n}" for i, n in enumerate(class_names)]))


# -------------------- LIVE --------------------
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        cfg = cfg_get()
        try:
            processed, _, _ = process_frame(
                img, sess, input_name, imgsz, class_names,
                cfg["conf_ppe"], cfg["conf_person"], cfg["strict"], cfg["draw_counts"]
            )
        except Exception as e:
            # чтобы LIVE не падал насмерть
            cv2.putText(img, f"ERROR: {str(e)[:80]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            processed = img
        return av.VideoFrame.from_ndarray(processed, format="bgr24")


tab1, tab2 = st.tabs(["🎥 LIVE ВИДЕО", "📁 АНАЛИЗ ФОТО"])

with tab1:
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
        res, safe, danger = process_frame(
            img_bgr, sess, input_name, imgsz, class_names,
            conf_ppe, conf_person, strict, draw_counts=False  # на фото счётчик НЕ рисуем
        )
        st.image(cv2.cvtColor(res, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.write(f"### SAFE: {safe} | DANGER: {danger}")
