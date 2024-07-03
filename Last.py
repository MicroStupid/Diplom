# %%
import cv2
import time
import numpy as np

# Глобальные переменные
drawing = False  # флаг рисования
roi = []  # область интереса

line_height = 100  # начальная высота линии от низа экрана

# Новые переменные для порогов HSV
lower_hsv = np.array([0, 0, 40])  # Нижний порог HSV
upper_hsv = np.array([180, 50, 220])  # Верхний порог HSV

binary_threshold = 40 # порог бинаризации
current_mouse_position = (0, 0)  # текущая позиция курсора

contour_crossing_start = None  # Время начала пересечения контура с линией
signal_start_time = None  # Время начала сигнала
signal_end_time = None

crossing_time = 0.65
buffer_duration = 0.4 # Длительность буфера в секундах
last_contour_time = None  # Время последнего обнаружения контура в области интереса

def on_line_height_change(trackbarValue):
    global line_height, contour_crossing_start, signal_start_time
    line_height = trackbarValue
    contour_crossing_start = None  # Сброс времени начала пересечения
    signal_start_time = None  # Сброс времени начала сигнала
    

def draw_rectangle(event, x, y, flags, param):
    global roi, drawing, current_mouse_position

    current_mouse_position = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi = [x, y, x, y]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi[2] = x
            roi[3] = y
    elif event == cv2.EVENT_LBUTTONUP:                                 
        drawing = False
        roi[2] = x          
        roi[3] = y


def detect_and_draw_contour(frame):
        # Проверяем, задана ли область интереса (ROI)
    if roi and len(roi) == 4:
        # Вырезаем ROI из кадра
        x1, y1, x2, y2 = roi
        frame_roi = frame[y1:y2, x1:x2]
        # Дополнительная проверка, чтобы убедиться, что координаты ROI корректны
        if x1 < x2 and y1 < y2:
            frame_roi = frame[y1:y2, x1:x2]
        else:
            frame_roi = None
    else:
        frame_roi = frame
    
    # Проверяем, не пуст ли frame_roi перед применением фильтра Гаусса
    if frame_roi is not None and frame_roi.size > 0:
    # Применяем фильтр Гаусса для сглаживания изображения
        frame_blurred = cv2.GaussianBlur(frame_roi, (5, 5), 0)
        hsv = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        #gray = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV)
        # Динамическая корректировка порога бинаризации
        #mean_intensity = np.mean(gray)
        #dynamic_threshold = binary_threshold * (1 + (mean_intensity - 128) / 128 * 0.95)
        #_, binary = cv2.threshold(gray, dynamic_threshold, 255, cv2.THRESH_TOZERO_INV)
    
        # Морфологические операции
        #kernel = np.ones((7, 7), np.uint8)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            valid_contours = [c for c in contours if cv2.contourArea(c) > 1 and is_valid_contour(c)]
    
            if valid_contours:
                max_contour = max(valid_contours, key=cv2.contourArea)
                cv2.drawContours(frame_roi, [max_contour], -1, (0, 255, 0), 3)
                check_contour_intersection(max_contour, frame_roi)


def is_valid_contour(contour):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    # Предполагаем, что кучка будет иметь соотношение сторон ближе к горизонтальным
    return aspect_ratio > 1

def check_contour_intersection(contour, frame):
    global line_height, contour_crossing_start, signal_start_time, signal_end_time, last_contour_time
    # Получаем координаты ограничивающего прямоугольника контура
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    # Вычисляем позицию линии относительно низа экрана
    line_position_from_bottom = frame.shape[0] - line_height
    # Проверяем, пересекает ли верхняя граница контура линию
    if y < line_position_from_bottom < y+h:
        last_contour_time = time.time()  # Обновляем время последнего обнаружения контура
        if contour_crossing_start is None:
            contour_crossing_start = time.time()  # Начало пересечения
        # Проверяем, находится ли контур в области интереса достаточно долго
        if time.time() - contour_crossing_start >= crossing_time:
            signal_start_time = True  # Сигнал активирован
            signal_end_time = time.time()  # Обновляем время окончания сигнала
    else:
        if last_contour_time and (time.time() - last_contour_time >= buffer_duration):
            # Деактивируем сигнал только если прошло достаточно времени после последнего обнаружения контура
            contour_crossing_start = None
            signal_start_time = False
            signal_end_time = None
            last_contour_time = None  # Сбрасываем время последнего обнаружения контура
        
    # Отображаем сигнал на экране, если он активен и длится достаточно долго
    if signal_start_time and signal_end_time:
        text = "Signal issued!"
        cv2.putText(frame, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else: 
        text = "No Signal!"
        cv2.putText(frame, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        

cap = cv2.VideoCapture("./Try2.mp4")  # Используйте 0 для локальной каме

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", draw_rectangle)

# Получение размеров кадра для установки максимального значения ползунка
ret, frame = cap.read()
if ret:
    cv2.createTrackbar("Line Height", "Frame", line_height, frame.shape[0], on_line_height_change)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if roi:
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 2)

    detect_and_draw_contour(frame)

    # Рисование линии с учетом ее положения от низа экрана
    cv2.line(frame, (0, frame.shape[0] - line_height), (frame.shape[1], frame.shape[0] - line_height), (255, 0, 0), 2)
    cv2.putText(frame, f"Cursor: {current_mouse_position}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    time.sleep(0.03)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == ord("r"):  # Сброс ROI
        roi = []

cap.release()
cv2.destroyAllWindows()


