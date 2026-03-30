import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import threading

# Настройка страницы (должна быть первой st-командой)
st.set_page_config(page_title="SafeGuard LIVE", layout="centered")
st.title("🛡️ SafeGuard ИИ: Real-Time")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# -------------------- Загрузка модели --------------------
@st.cache_resource
def load_model():
    return YOLO("best.onnx", task="detect")

try:
    model = load_model()
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

# -------------------- UI: настройки --------------------
all_classes = [str(v) for v in model.names.values()]

st.sidebar.write("### Обнаружение классов модели:")
st.sidebar.write(all_classes)

conf_val = st.sidebar.slider("Чувствительность (общая)", 0.05, 1.0, 0.40, 0.05)

st.sidebar.write("---")
st.sidebar.write("### Настройка логики СИЗ")

# Выбор класса человека
default_person = all_classes.index("person") if "person" in all_classes else 0
person_class = st.sidebar.selectbox("Класс человека", all_classes, index=default_person)

# Выбор классов, которые считаем СИЗ
default_ppe = [c for c in ["helmet", "vest"] if c in all_classes]
ppe_classes = st.sidebar.multiselect("Классы СИЗ (что считаем защитой)", all_classes, default=default_ppe)

# Выбор прямых нарушений (если в модели есть)
default_viol = [c for c in ["no-helmet", "no-vest"] if c in all_classes]
violation_classes = st.sidebar.multiselect("Классы прямых нарушений (если есть)", all_classes, default=default_viol)

# Тонкость: проверка попадания центра СИЗ в человека лучше, чем любое пересечение
use_center = st.sidebar.checkbox("Привязка СИЗ по центру (рекомендую)", value=True)

# Показывать подписи на СИЗ
show_ppe_labels = st.sidebar.checkbox("Подписывать найденные СИЗ", value=True)

# -------------------- Синхронизация настроек для LIVE потока --------------------
_cfg_lock = threading.Lock()
CFG = {
    "conf": conf_val,
    "person_class": person_class,
    "ppe_classes": ppe_classes,
    "violation_classes": violation_classes,
    "use_center": use_center,
    "show_ppe_labels": show_ppe_labels,
}

def cfg_set(**kwargs):
    with _cfg_lock:
        CFG.update(kwargs)

def cfg_get():
    with _cfg_lock:
        return dict(CFG)

cfg_set(
    conf=conf_val,
    person_class=person_class,
    ppe_classes=ppe_classes,
    violation_classes=violation_classes,
    use_center=use_center,
    show_ppe_labels=show_ppe_labels,
)

# -------------------- Рисование кружков-счётчиков --------------------
def draw_counter_circles(img, safe_count: int, danger_count: int):
    # Зеленый круг SAFE
    cv2.circle(img, (30, 30), 18, (0, 200, 0), -1)
    cv2.putText(img, str(safe_count), (23, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Красный круг DANGER
    cv2.circle(img, (30, 75), 18, (0, 0, 255), -1)
    cv2.putText(img, str(danger_count), (23, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Подписи мелким (по желанию можно убрать)
    cv2.putText(img, "SAFE", (55, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
    cv2.putText(img, "DANGER", (55, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


# -------------------- Основная логика обработки --------------------
def process_image_logic(img_cv, model, cfg):
    conf = cfg["conf"]
    person_class_name = cfg["person_class"]
    ppe_classes = set([c.lower() for c in cfg["ppe_classes"]])
    violation_classes = set([c.lower() for c in cfg["violation_classes"]])
    use_center = cfg["use_center"]
    show_ppe_labels = cfg["show_ppe_labels"]

    # Предсказание
    results = model.predict(img_cv, conf=conf, verbose=False)
    boxes = results[0].boxes

    people = []
    ppe_boxes = []
    viol_boxes = []

    # Собираем боксы по группам
    for box in boxes:
        cls_id = int(box.cls[0])
        label = str(model.names[cls_id]).lower()
        coords = box.xyxy[0].tolist()
        score = float(box.conf[0])

        if label == person_class_name.lower():
            people.append(coords)
        elif label in violation_classes:
            viol_boxes.append({"label": label, "coords": coords, "score": score})
        elif label in ppe_classes:
            ppe_boxes.append({"label": label, "coords": coords, "score": score})
        else:
            # остальные классы игнорируем
            pass

    # Рисуем PPE зелёным
    for p in ppe_boxes:
        x1, y1, x2, y2 = map(int, p["coords"])
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 200, 0), 2)
        if show_ppe_labels:
            cv2.putText(
                img_cv,
                f"{p['label']} {p['score']:.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 0),
                2,
            )

    # Рисуем прямые нарушения красным
    for v in viol_boxes:
        x1, y1, x2, y2 = map(int, v["coords"])
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            img_cv,
            f"{v['label']} {v['score']:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )

    safe_count = 0
    danger_count = 0

    # Проверка людей
    for p in people:
        px1, py1, px2, py2 = p

        # 1) если есть прямое нарушение, пересекающееся с человеком -> DANGER
        has_direct_violation = any(
            not (v["coords"][2] < px1 or v["coords"][0] > px2 or v["coords"][3] < py1 or v["coords"][1] > py2)
            for v in viol_boxes
        )
        if has_direct_violation:
            danger_count += 1
            cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 3)
            cv2.putText(
                img_cv,
                "NO PPE",
                (int(px1), max(0, int(py1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            continue

        # 2) иначе — проверяем наличие СИЗ
        is_protected = False
        if use_center:
            # Центр СИЗ должен попадать в бокс человека
            for prot in ppe_boxes:
                rx1, ry1, rx2, ry2 = prot["coords"]
                cx = (rx1 + rx2) / 2.0
                cy = (ry1 + ry2) / 2.0
                if px1 <= cx <= px2 and py1 <= cy <= py2:
                    is_protected = True
                    break
        else:
            # Любое пересечение
            for prot in ppe_boxes:
                rx1, ry1, rx2, ry2 = prot["coords"]
                if not (rx2 < px1 or rx1 > px2 or ry2 < py1 or ry1 > py2):
                    is_protected = True
                    break

        if is_protected:
            safe_count += 1
            cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (0, 255, 0), 3)
            cv2.putText(
                img_cv,
                "PPE OK",
                (int(px1), max(0, int(py1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            danger_count += 1
            cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 3)
            cv2.putText(
                img_cv,
                "NO PPE",
                (int(px1), max(0, int(py1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

    # Рисуем кружки-счётчики
    draw_counter_circles(img_cv, safe_count, danger_count)

    return img_cv, safe_count, danger_count


# -------------------- LIVE Video processor --------------------
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        cfg = cfg_get()
        processed_img, _, _ = process_image_logic(img, model, cfg)
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")


# -------------------- Интерфейс --------------------
tab1, tab2 = st.tabs(["🎥 Живое видео", "📁 Загрузить фото"])

with tab1:
    st.write("Нажмите **Start** для запуска мониторинга.")
    ctx = webrtc_streamer(
        key="ppe-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Показать числа под видео (если доступно)
    st.caption("Счётчики также рисуются кружками на видео в левом верхнем углу.")

with tab2:
    up_img = st.file_uploader("Выберите фото", type=["jpg", "png", "jpeg"])
    if up_img:
        img = Image.open(up_img)
        img_cv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

        cfg = cfg_get()
        res_cv, safe, danger = process_image_logic(img_cv, model, cfg)

        st.image(cv2.cvtColor(res_cv, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.markdown(
            f"""
            <div style="display:flex; gap:16px; align-items:center;">
              <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:14px;height:14px;border-radius:50%;background:#22c55e;"></div>
                <div><b>В защите:</b> {safe}</div>
              </div>
              <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:14px;height:14px;border-radius:50%;background:#ef4444;"></div>
                <div><b>Без защиты:</b> {danger}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
