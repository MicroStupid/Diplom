# %%
import cv2
import numpy as np
import time

# Определение области интереса (ROI)
roi = None
drawing = False
selection_time = 0
threshold = 25  # Макс граница заполненности
current_mouse_position = (0, 0)  # Текущая позиция курсора

# Загрузка шрифта TTF для русского языка
fontpath = "./ofont.ru_Nunito.ttf"  # Укажите путь к файлу шрифта TTF
font = cv2.FONT_HERSHEY_SIMPLEX


def draw_rectangle(event, x, y, flags, param):
    global roi, drawing, selection_time, current_mouse_position

    current_mouse_position = (x, y)  # Обновление текущей позиции курсора

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi = (x, y, 0, 0)
        selection_time = time.time()
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi = (roi[0], roi[1], x - roi[0], y - roi[1])
        selection_time = time.time() - selection_time
        
    # Координаты квадрата
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.putText(frame, f"({x}, {y})", (x, y), font, 0.5, (0, 0, 255), 2)


min_contour_area = 18000  # Минимальная площадь контура
max_contour_area = 50000  # Максимальная площадь контура

def detect_fill_level(frame):
    
      # Проверяем, выделена ли область интереса (ROI)
    if roi is not None and roi[2] > 0 and roi[3] > 0:  # Проверяем, что ширина и высота ROI больше 0
        # Обрезаем кадр до ROI
        x, y, w, h = roi
        frame_roi = frame[y:y+h, x:x+w]
    else:
        frame_roi = frame
    
    # Преобразование изображения в оттенки серого
    gray_frame = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    
    # Бинаризация изображения
    _, binary_frame = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    
    # Инвертирование бинарного изображения
    inverted_frame = cv2.bitwise_not(binary_frame)
    
    # Поиск контуров на инвертированном изображении
    contours, _ = cv2.findContours(inverted_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Фильтрация контуров по площади
    filtered_contours = [contour for contour in contours if min_contour_area <= cv2.contourArea(contour) <= max_contour_area]
    
    # Выбор контура с наибольшей площадью из отфильтрованных
    max_contour = None
    max_area = 0
    for contour in filtered_contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_contour = contour
            max_area = area
    
     # Отображение контура и возвращение количества серых пикселей внутри контура, если таковой найден
    if max_contour is not None:
        # Отображение контура
        if roi is not None and roi[2] > 0 and roi[3] > 0:
            cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2, offset=(x,y))  # Добавляем смещение для ROI
        else:
            cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)
        
        # Возвращаем количество серых пикселей внутри контура
        mask = np.zeros_like(inverted_frame)
        cv2.drawContours(mask, [max_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        gray_pixels = cv2.countNonZero(cv2.bitwise_and(mask, inverted_frame))
        return gray_pixels
    else:
        return 0


# Функция для выдачи сигнала, если уровень заполненности превышает определенную отметку
def issue_signal(fill_level, frame):
    if fill_level > threshold:
        start_time = time.time()
        while fill_level > threshold:
            cv2.putText(frame, "Moving", (frame.shape[1]-150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Container Fill Level', frame)
            cv2.waitKey(10)
            ret, frame = cap.read()
            fill_level = detect_fill_level(frame)
        end_time = time.time() - start_time
        print(start_time, end_time)
        print("Signal finished!") 
        
    

# Открытие видеопотока с камеры
cap = cv2.VideoCapture("./SCHOM12.mp4")

# Рассчитываем задержку для воспроизведения с реальной скоростью
fps = cap.get(cv2.CAP_PROP_FPS)  # Получаем FPS видео
delay = int(1000 / fps)  # Рассчитываем задержку в мс

cv2.namedWindow('Container Fill Level')
cv2.setMouseCallback('Container Fill Level', draw_rectangle)

# Создание ползунка для регулировки threshold
def on_threshold_change(value):
    global threshold
    threshold = value

cv2.createTrackbar('Threshold', 'Container Fill Level', threshold, 255, on_threshold_change)

# Чтение первого кадра из видеопотока
ret, frame = cap.read()

# Определение начального значения prev_fill_level
prev_fill_level = detect_fill_level(frame)

while True:
    # Передача кадра функции для определения уровня заполненности
    fill_level = detect_fill_level(frame)


    # Выдача сигнала, если уровень заполненности превышает отметку
    if fill_level > threshold:
        issue_signal(fill_level, frame)

    # Отображение выделенной зоны
    if roi is not None:
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 2)
         
    # Отображение координат курсора
    cv2.putText(frame, f"({current_mouse_position[0]}, {current_mouse_position[1]})", current_mouse_position, font, 0.5, (0, 0, 255), 2)

    cv2.imshow('Container Fill Level', frame)

    # Выход из цикла при нажатии клавиши 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # Чтение следующего кадра из видеопотока
    ret, frame = cap.read()
    cv2.waitKey(10)


# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()


