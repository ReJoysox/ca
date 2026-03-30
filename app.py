# --- ФУНКЦИЯ ОБРАБОТКИ КАДРА ---
def process_image_logic(img_cv, model, conf):
    results = model.predict(img_cv, conf=conf, verbose=False)
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        return img_cv

    people = []
    protection_boxes = []

    for box in boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id].lower()
        coords = box.xyxy[0].tolist()
        
        if 'person' in label or 'human' in label:
            people.append(coords)
        else:
            protection_boxes.append(coords)
            cv2.rectangle(img_cv, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (0, 255, 0), 2)
            cv2.putText(img_cv, label.upper(), (int(coords[0]), int(coords[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    for p in people:
        px1, py1, px2, py2 = p
        is_protected = False
        for prot in protection_boxes:
            rx1, ry1, rx2, ry2 = prot
            if not (rx2 < px1 or rx1 > px2 or ry2 < py1 or ry1 > py2):
                is_protected = True
                break
        
        if is_protected:
            cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (255, 255, 255), 1)
        else:
            cv2.putText(img_cv, "NO PROTECTION", (int(px1), int(py1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            head_h = int((py2 - py1) * 0.25)
            cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py1 + head_h)), (0, 0, 255), 3)
            cv2.rectangle(img_cv, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 1)
    
    return img_cv

# --- КЛАСС ДЛЯ REAL-TIME ВИДЕО ---
class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Обрабатываем кадр
        processed_img = process_image_logic(img, model, conf_val)
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# Интерфейс
tab1, tab2 = st.tabs(["🎥 Живое видео", "📁 Загрузить фото"])

with tab1:
    st.write("Нажмите 'Start' для запуска мониторинга")
    webrtc_streamer(
        key="ppe-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )

with tab2:
    up_img = st.file_uploader("Выберите фото", type=['jpg', 'png', 'jpeg'])
    if up_img:
        img = Image.open(up_img)
        img_cv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        res_cv = process_image_logic(img_cv, model, conf_val)
        st.image(cv2.cvtColor(res_cv, cv2.COLOR_BGR2RGB), width=500)
