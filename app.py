import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Настройка страницы
st.set_page_config(page_title="SafeGuard AI", layout="centered")
st.title("🛡️ SafeGuard ИИ: Контроль СИЗ")

# Загрузка модели
@st.cache_resource
def load_model():
    try:
        return YOLO('best.onnx', task='detect')
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None

model = load_model()

# --- Вспомогательные математические функции ---
def compute_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top: return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return intersection_area / float(box1_area + box2_area - intersection_area)

def is_ppe_on_person(person_box, ppe_box):
    x_left = max(person_box[0], ppe_box[0])
    y_top = max(person_box[1], ppe_box[1])
    x_right = min(person_box[2], ppe_box[2])
    y_bottom = min(person_box[3], ppe_box[3])

    if x_right < x_left or y_bottom < y_top: return False

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    ppe_area = (ppe_box[2] - ppe_box[0]) * (ppe_box[3] - ppe_box[1])
    
    return ppe_area > 0 and (intersection_area / ppe_area) > 0.3

# --- Основная логика программы ---
if model:
    all_classes = [name.lower() for name in model.names.values()]
    person_classes = [c for c in all_classes if 'person' in c or 'human' in c]
    ppe_classes = [c for c in all_classes if c not in person_classes]
    helmet_keywords = ['helmet', 'hardhat', 'hard_hat', 'каска']
    default_helmets = [c for c in ppe_classes if any(keyword in c for keyword in helmet_keywords)] or ppe_classes

    st.sidebar.write("### ⚙️ Настройки детекции")
    conf_val = st.sidebar.slider("Чувствительность", 0.05, 1.0, 0.25, 0.05)
    selected_ppe = st.sidebar.multiselect("Целевые СИЗ:", options=ppe_classes, default=default_helmets)

    def process_image_logic(img_cv, model, conf, target_ppe):
        # Используем augment=True для максимальной точности поиска касок
        results = model.predict(img_cv, conf=conf, augment=True, agnostic_nms=False, verbose=False)
        boxes = results[0].boxes
        
        protected_count, unprotected_count = 0, 0
        if len(boxes) == 0: return img_cv, protected_count, unprotected_count

        raw_people = []
        protection_boxes = []

        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id].lower()
            coords = box.xyxy[0].tolist()
            
            if label in person_classes:
                raw_people.append({"coords": coords, "conf": float(box.conf[0])})
            elif label in target_ppe:
                # МЫ ТОЛЬКО СОХРАНЯЕМ КООРДИНАТЫ КАСКИ, НО НЕ РИСУЕМ ЕЕ
                protection_boxes.append(coords)

        # Удаление дубликатов людей
        raw_people.sort(key=lambda x: x['conf'], reverse=True)
        filtered_people = []
        for person in raw_people:
            if not any(compute_iou(person["coords"], fp["coords"]) > 0.4 for fp in filtered_people):
                filtered_people.append(person)

        # Отрисовка только людей
        for p in filtered_people:
            px1, py1, px2, py2 = p["coords"]
            is_protected = any(is_ppe_on_person(p["coords"], prot_coords) for prot_coords in protection_boxes)
            
            if is_protected:
                protected_count += 1
                # Только зеленая рамка вокруг человека
                cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (0, 255, 0), 3)
                cv2.putText(img_cv, "OK", (int(px1), int(py1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                unprotected_count += 1
                # Только красная рамка вокруг человека
                cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 3)
                cv2.putText(img_cv, "NO PPE", (int(px1), int(py1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return img_cv, protected_count, unprotected_count

    # --- ИНТЕРФЕЙС ---
    st.write("### 📁 Загрузите фото для анализа")
    up_img = st.file_uploader("", type=['jpg', 'png', 'jpeg'])
    
    if up_img:
        img = Image.open(up_img)
        img_cv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        
        with st.spinner('Анализирую изображение...'):
            res_cv, p_count, u_count = process_image_logic(img_cv, model, conf_val, selected_ppe)
        
        st.image(cv2.cvtColor(res_cv, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>📊 Результат проверки:</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"### 🟢 В СИЗ: **{p_count}**")
        with col2:
            st.error(f"### 🔴 Без СИЗ: **{u_count}**")

else:
    st.error("Ошибка загрузки модели `best.onnx`.")
