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

# Вспомогательные функции
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
    
    return ppe_area > 0 and (intersection_area / ppe_area) > 0.5

if model:
    all_classes = [name.lower() for name in model.names.values()]
    person_classes = [c for c in all_classes if 'person' in c or 'human' in c]
    ppe_classes = [c for c in all_classes if c not in person_classes]

    st.sidebar.write("### Настройки детекции")
    conf_val = st.sidebar.slider("Чувствительность", 0.1, 1.0, 0.3)
    
    selected_ppe = st.sidebar.multiselect(
        "Выберите СИЗ для контроля:",
        options=ppe_classes,
        default=ppe_classes
    )

    # --- ФУНКЦИЯ ОБРАБОТКИ КАДРА ---
    def process_image_logic(img_cv, model, conf, target_ppe):
        results = model.predict(img_cv, conf=conf, verbose=False)
        boxes = results[0].boxes
        
        protected_count = 0
        unprotected_count = 0
        
        if len(boxes) == 0:
            return img_cv, protected_count, unprotected_count

        raw_people = []
        protection_boxes = []

        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id].lower()
            coords = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            
            if label in person_classes:
                raw_people.append({"coords": coords, "conf": confidence})
            elif label in target_ppe:
                protection_boxes.append(coords)
                cv2.rectangle(img_cv, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (255, 255, 0), 2)
                cv2.putText(img_cv, label.upper(), (int(coords[0]), int(coords[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        raw_people.sort(key=lambda x: x['conf'], reverse=True)
        filtered_people = []
        
        for person in raw_people:
            if not any(compute_iou(person["coords"], fp["coords"]) > 0.4 for fp in filtered_people):
                filtered_people.append(person)

        for p in filtered_people:
            px1, py1, px2, py2 = p["coords"]
            is_protected = any(is_ppe_on_person(p["coords"], prot_coords) for prot_coords in protection_boxes)
            
            if is_protected:
                protected_count += 1
                cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (0, 255, 0), 2)
                cv2.putText(img_cv, "PROTECTED", (int(px1), int(py1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                unprotected_count += 1
                cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 2)
                cv2.putText(img_cv, "NO PROTECTION", (int(px1), int(py1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return img_cv, protected_count, unprotected_count

    # --- ИНТЕРФЕЙС ---
    tab1, tab2 = st.tabs(["🎥 Живое видео", "📁 Загрузить фото"])

    with tab1:
        st.write("### Выберите метод захвата камеры:")
        cam_mode = st.radio("Метод:", ["Локальная камера (100% работает)", "WebRTC (Для браузера/сервера)"])
        
        if cam_mode == "Локальная камера (100% работает)":
            st.info("Этот метод запускает камеру напрямую через Python. Отлично подходит для запуска на своем ПК.")
            run_camera = st.checkbox("🟢 Включить камеру")
            FRAME_WINDOW = st.image([])
            
            if run_camera:
                cap = cv2.VideoCapture(0) # 0 - стандартная веб-камера
                while run_camera:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Не удалось получить доступ к камере.")
                        break
                    
                    # Обработка
                    processed_img, p_count, u_count = process_image_logic(frame, model, conf_val, selected_ppe)
                    
                    # Рисуем счетчики
                    cv2.putText(processed_img, f"Protected: {p_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(processed_img, f"Unprotected: {u_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Вывод в Streamlit (конвертация BGR -> RGB)
                    FRAME_WINDOW.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
                cap.release()
                
        else:
            st.info("Метод WebRTC. Нажмите Start. Если видео не грузится, используйте 'Локальную камеру'.")
            
            # Современный метод для WebRTC (без классов)
            def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                
                # Защита от огромных разрешений
                h, w = img.shape[:2]
                if w > 640:
                    img = cv2.resize(img, (640, int(640 * h / w)))

                processed_img, p_count, u_count = process_image_logic(img, model, conf_val, selected_ppe)
                
                cv2.putText(processed_img, f"Protected: {p_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(processed_img, f"Unprotected: {u_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

            webrtc_streamer(
                key="ppe-stream",
                video_frame_callback=video_frame_callback,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )

    with tab2:
        up_img = st.file_uploader("Выберите фото", type=['jpg', 'png', 'jpeg'])
        if up_img:
            img = Image.open(up_img)
            img_cv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
            
            res_cv, p_count, u_count = process_image_logic(img_cv, model, conf_val, selected_ppe)
            
            st.image(cv2.cvtColor(res_cv, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            st.markdown("---")
            st.markdown("<h3 style='text-align: center;'>Статистика распознавания:</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### 🟢 В СИЗ (Есть защита): **{p_count}**")
            with col2:
                st.markdown(f"### 🔴 Без СИЗ (Нет защиты): **{u_count}**")

else:
    st.error("Ошибка загрузки модели.")
