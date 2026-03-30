import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av

# Настройка страницы
st.set_page_config(page_title="SafeGuard LIVE", layout="centered")
st.title("🛡️ SafeGuard ИИ: Real-Time")

# Загрузка модели
@st.cache_resource
def load_model():
    try:
        return YOLO('best.onnx', task='detect')
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None

model = load_model()

if model:
    # Получаем все классы из модели и приводим к нижнему регистру
    all_classes = [name.lower() for name in model.names.values()]
    
    # Отделяем класс человека от классов СИЗ
    person_classes = [c for c in all_classes if 'person' in c or 'human' in c]
    ppe_classes = [c for c in all_classes if c not in person_classes]

    st.sidebar.write("### Настройки детекции")
    conf_val = st.sidebar.slider("Чувствительность модели", 0.1, 1.0, 0.4)
    
    # Поле выбора конкретных СИЗ для контроля
    selected_ppe = st.sidebar.multiselect(
        "Выберите классы СИЗ для проверки защиты (например, только каски):",
        options=ppe_classes,
        default=ppe_classes, # По умолчанию выбраны все доступные СИЗ
        help="Человек будет считаться защищенным, только если на нем надет выбранный элемент."
    )

    # --- ФУНКЦИЯ ОБРАБОТКИ КАДРА ---
    def process_image_logic(img_cv, model, conf, target_ppe):
        results = model.predict(img_cv, conf=conf, verbose=False)
        boxes = results[0].boxes
        
        protected_count = 0
        unprotected_count = 0
        
        if len(boxes) == 0:
            return img_cv, protected_count, unprotected_count

        people = []
        protection_boxes = []

        # Распределяем найденные объекты
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id].lower()
            coords = box.xyxy[0].tolist()
            
            if label in person_classes:
                people.append(coords)
            elif label in target_ppe: # Учитываем только выбранные в фильтре классы
                protection_boxes.append((coords, label))
                # Отрисовка рамки самого СИЗ (например, каски) - синим цветом
                cv2.rectangle(img_cv, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (255, 255, 0), 2)
                cv2.putText(img_cv, label.upper(), (int(coords[0]), int(coords[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Проверка защиты для каждого человека
        for p in people:
            px1, py1, px2, py2 = p
            is_protected = False
            
            # Проверяем, пересекается ли человек с выбранными СИЗ
            for prot_coords, prot_label in protection_boxes:
                rx1, ry1, rx2, ry2 = prot_coords
                # Логика пересечения прямоугольников
                if not (rx2 < px1 or rx1 > px2 or ry2 < py1 or ry1 > py2):
                    is_protected = True
                    break
            
            if is_protected:
                protected_count += 1
                # Зеленый квадрат для человека В ЗАЩИТЕ
                cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (0, 255, 0), 2)
                cv2.putText(img_cv, "PROTECTED", (int(px1), int(py1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                unprotected_count += 1
                # Красный квадрат для человека БЕЗ ЗАЩИТЫ
                cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 2)
                cv2.putText(img_cv, "NO PROTECTION", (int(px1), int(py1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return img_cv, protected_count, unprotected_count

    # --- КЛАСС ДЛЯ REAL-TIME ВИДЕО ---
    class VideoProcessor:
        def __init__(self):
            self.conf = 0.4
            self.target_ppe = []

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            # Обрабатываем кадр с использованием динамических параметров
            processed_img, p_count, u_count = process_image_logic(img, model, self.conf, self.target_ppe)
            
            # Счетчики в левом верхнем углу видео
            cv2.putText(processed_img, f"Protected: {p_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(processed_img, f"Unprotected: {u_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

    # Интерфейс
    tab1, tab2 = st.tabs(["🎥 Живое видео", "📁 Загрузить фото"])

    with tab1:
        st.write("Нажмите 'Start' для запуска мониторинга камеры.")
        
        ctx = webrtc_streamer(
            key="ppe-detection",
            video_processor_factory=VideoProcessor,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }
        )
        
        # Обновление параметров для живого видео на лету
        if ctx.video_processor:
            ctx.video_processor.conf = conf_val
            ctx.video_processor.target_ppe = selected_ppe

    with tab2:
        up_img = st.file_uploader("Выберите фото", type=['jpg', 'png', 'jpeg'])
        if up_img:
            img = Image.open(up_img)
            img_cv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
            
            # Обработка с учетом выбранных классов СИЗ
            res_cv, p_count, u_count = process_image_logic(img_cv, model, conf_val, selected_ppe)
            
            st.image(cv2.cvtColor(res_cv, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Статистика
            st.markdown("---")
            st.markdown("<h3 style='text-align: center;'>Статистика распознавания:</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### 🟢 В СИЗ (Есть защита): **{p_count}**")
            with col2:
                st.markdown(f"### 🔴 Без СИЗ (Нет защиты): **{u_count}**")

else:
    st.error("Ошибка загрузки модели.")
