import os
os.environ["ORT_SKIP_OPSET_VALIDATION"] = "1"

from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image


# ---------------- Streamlit ----------------
st.set_page_config(page_title="SafeGuard PRO", layout="centered")
st.title("SafeGuard PRO — Фото анализ СИЗ")


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "best.onnx"
CLASSES_PATH = ROOT / "classes.txt"


@st.cache_resource
def load_classes():
    return [x.strip() for x in CLASSES_PATH.read_text().splitlines()]


@st.cache_resource
def load_model():
    sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    return sess, input_name


def norm(s):
    return s.lower().replace("_", "-")


def process_image(img, sess, input_name, classes, conf):
    h0, w0 = img.shape[:2]

    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = cv2.resize(x, (640, 640))
    x = np.transpose(x, (2, 0, 1))[None, ...]

    out = sess.run(None, {input_name: x})[0]

    if out.ndim == 3:
        out = out[0]

    if out.shape[1] != 6:
        st.error("ONNX формат не Nx6. Нужен x1,y1,x2,y2,score,class")
        return img, 0, 0

    safe = 0
    danger = 0

    for det in out:
        x1, y1, x2, y2, score, cls_id = det
        if score < conf:
            continue

        label = classes[int(cls_id)]

        x1 = int(x1 * w0 / 640)
        x2 = int(x2 * w0 / 640)
        y1 = int(y1 * h0 / 640)
        y2 = int(y2 * h0 / 640)

        l = norm(label)

        if l == "person":
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

        elif l == "helmet" or l == "vest":
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            safe += 1

        elif l == "no-helmet" or l == "no-vest":
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            danger += 1

    return img, safe, danger


try:
    classes = load_classes()
    sess, input_name = load_model()
except Exception as e:
    st.error(f"Ошибка загрузки: {e}")
    st.stop()


conf_val = st.slider("Чувствительность", 0.05, 1.0, 0.3, 0.05)

uploaded = st.file_uploader("Загрузите фото", type=["jpg", "jpeg", "png"])

if uploaded:
    img = cv2.cvtColor(np.array(Image.open(uploaded).convert("RGB")), cv2.COLOR_RGB2BGR)
    result, safe, danger = process_image(img, sess, input_name, classes, conf_val)

    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.write(f"SAFE: {safe} | DANGER: {danger}")
