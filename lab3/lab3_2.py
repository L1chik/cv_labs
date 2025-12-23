#:Lab3_2 Arithmetic Operations on Images
import cv2 as cv
import numpy as np
import os

# 1. Загрузка изображений
img1 = cv.imread('./img2.jpg')
img2 = cv.imread('../lab2/open-cv-logo.jpg')

assert img1 is not None, "file could not be read, check with os.path.exists()"
assert img2 is not None, "file could not be read, check with os.path.exists()"

# Убедимся, что изображения одного размера
h, w = img1.shape[:2]
img2 = cv.resize(img2, (w, h))

# 2. Сложение изображений (с насыщением)
added = cv.add(img1, img2)
cv.imshow('Added (cv.add)', added)

# 3. Смешивание изображений
alpha = 0.7
beta = 0.3
gamma = 0
blended = cv.addWeighted(img1, alpha, img2, beta, gamma)
cv.imshow('Blended (addWeighted)', blended)

# 4. Вычитание изображений
subtracted = cv.subtract(img1, img2)
cv.imshow('Subtracted', subtracted)

# 5. Добавляем лого поверх изображения
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols]

# Now create a mask of logo and create its inverse mask also
img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)

# Now black-out the area of logo in ROI
img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)

# Take only region of logo from logo image.
img2_fg = cv.bitwise_and(img2,img2,mask = mask)

# Put logo in ROI and modify the main image
dst = cv.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst

cv.imshow('res',img1)

cv.waitKey(0)
cv.destroyAllWindows()

#===========================Доп задание==============================
# Путь к папке с изображениями
folder = "./images"   # замени на свою папку
delay = 2           # время показа каждого кадра (в секундах)
transition_time = 1 # длительность перехода (в секундах)
fps = 30            # кадров в секунду для плавности

# Список файлов изображений
files = sorted([
    os.path.join(folder, f)
    for f in os.listdir(folder)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

if len(files) < 2:
    raise ValueError("В папке должно быть минимум два изображения.")

# Загрузка
images = [cv.imread(f) for f in files]
h, w = images[0].shape[:2]

for i in range(len(images)):
    images[i] = cv.resize(images[i], (h, w))

# Показ слайдов
for i in range(len(images)):
    img1 = images[i]
    img2 = images[(i + 1) % len(images)]  # переход к следующему (по кругу)

    # Отображаем текущее изображение
    cv.imshow("Слайд-шоу", img1)
    if cv.waitKey(int(delay * 1000)) & 0xFF == 27:  # Esc — выход
        break

    # Плавный переход (перекрытие)
    for t in np.linspace(0, 1, int(fps * transition_time)):
        blended = cv.addWeighted(img1, 1 - t, img2, t, 0)
        cv.imshow("Слайд-шоу", blended)
        if cv.waitKey(int(1000 / fps)) & 0xFF == 27:
            break

cv.destroyAllWindows()