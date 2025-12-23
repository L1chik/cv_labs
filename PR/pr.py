import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# ============================================================
# Пути
# ============================================================

IMAGES_DIR = Path("PR/Images")
TEMPLATES_DIR = Path("PR/gosznak")
RESULTS_DIR = Path("PR/Results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Формат номера (8 символов)
# ============================================================

CHAR_PATTERN: List[str] = [
    "letter", "digit", "digit", "digit",
    "letter", "letter",
    "digit", "digit",
]

# Координаты нарезки символов из ROI (как в cv1)
CHAR_BOXES: List[Tuple[int, int, int, int]] = [
    (7, 4, 15, 20),
    (20, 0, 18, 30),
    (35, 3, 15, 21),
    (47, 0, 15, 30),
    (62, 5, 18, 19),
    (75, 5, 15, 19),
    (90, 0, 15, 20),
    (100, 0, 15, 20),
]


# ============================================================
# Коррекция яркости
# ============================================================

def adjust_brightness(image: np.ndarray) -> np.ndarray:
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


# ============================================================
# Поиск ROI номерного знака (адаптация cv1)
# ============================================================

def find_best_plate_roi(image: np.ndarray):
    filtered_img = cv2.GaussianBlur(image, (9, 9), 0)
    gray_image = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)

    threshold_values = list(range(50, 256, 5))

    best_roi = None
    best_solidity = 0.0
    best_rect = None
    best_M = None

    for thrs in threshold_values:
        _, binary = cv2.threshold(gray_image, thrs, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            rect = cv2.minAreaRect(contour)
            (x, y), (w, h), angle = rect

            if 15 < angle < 45:
                continue
            if w == 0 or h == 0:
                continue

            aspect_ratio = w / h if w > h else h / w
            area = w * h
            if not (2.5 < aspect_ratio < 5.8 and 1500 < area < 10000):
                continue

            contour_area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = contour_area / hull_area if hull_area > 0 else 0

            if solidity < 0.8:
                continue

            if solidity > best_solidity:
                best_solidity = solidity
                best_rect = rect

                angle_corr = angle if angle < 40 else angle - 90
                M = cv2.getRotationMatrix2D((x, y), angle_corr, 1.0)
                rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                best_M = M

                vert = h if w > h else w
                horz = w if w > h else h
                horz = min(horz, 120)

                y1 = int(y - vert / 2)
                y2 = int(y + vert / 2)
                x1 = int(x - horz / 2)
                x2 = int(x + horz / 2)

                H, W = image.shape[:2]
                y1, y2 = max(0, y1), min(H, y2)
                x1, x2 = max(0, x1), min(W, x2)

                best_roi = rotated[y1:y2, x1:x2]

    return best_roi, best_rect, best_M


# ============================================================
# Нарезка ROI на символы
# ============================================================

def split_number_by_image(image: np.ndarray) -> List[np.ndarray]:
    symbols: List[np.ndarray] = []
    H, W = image.shape[:2]

    for (x, y, w, h) in CHAR_BOXES:
        # защита от выхода за границы ROI
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))

        symbol = image[y:y + h, x:x + w]
        symbols.append(symbol)

    return symbols


# ============================================================
# Бинаризация символа
# ============================================================

def binaryzation_number_symbol(symbol_image: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(symbol_image, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.equalizeHist(grayscale)
    grayscale = cv2.GaussianBlur(grayscale, (3, 3), 0)

    binary_image = cv2.adaptiveThreshold(
        grayscale,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    return binary_image


# ============================================================
# Сравнение с шаблонами
# ============================================================

def compare_symbol_with_templates(symbol_bgr: np.ndarray,
                                  templates_folder: Path,
                                  is_digit: bool,
                                  is_region_digit: bool):
    """
    Возвращает:
      best_match: символ (строка)
      binary_symbol: бинаризованный символ (np.ndarray)
      best_template_img: изображение шаблона (np.ndarray)
    """
    if symbol_bgr is None or symbol_bgr.size == 0:
        return "?", None, None

    binary_symbol = binaryzation_number_symbol(symbol_bgr)
    sh, sw = binary_symbol.shape[:2]

    template_files = [
        f for f in os.listdir(templates_folder)
        if f.lower().endswith((".png", ".jpg"))
    ]

    if is_digit:
        template_files = [f for f in template_files if f[0].isdigit()]
    else:
        template_files = [f for f in template_files if not f[0].isdigit()]

    best_match = "?"
    best_score = -1.0
    best_template_img = None

    for tmpl_file in template_files:
        tmpl_img = cv2.imread(str(templates_folder / tmpl_file), cv2.IMREAD_GRAYSCALE)
        if tmpl_img is None:
            continue

        if is_region_digit:
            scale = 0.04285714285
        else:
            scale = 0.0488505 if is_digit else 0.0599078

        new_w = max(1, int(tmpl_img.shape[1] * scale))
        new_h = max(1, int(tmpl_img.shape[0] * scale))

        if new_h > sh or new_w > sw:
            ratio = min(sh / new_h, sw / new_w)
            new_w = max(1, int(new_w * ratio))
            new_h = max(1, int(new_h * ratio))

        if new_h < 1 or new_w < 1:
            continue

        tmpl_resized = cv2.resize(tmpl_img, (new_w, new_h))

        res = cv2.matchTemplate(binary_symbol, tmpl_resized, cv2.TM_CCOEFF_NORMED)
        score = cv2.minMaxLoc(res)[1]

        if score > best_score:
            best_score = score
            best_match = os.path.splitext(tmpl_file)[0]
            best_template_img = tmpl_resized

    return best_match, binary_symbol, best_template_img


# ============================================================
# Построение миниатюры символа: 150×150, 3 блока
# ============================================================

def make_symbol_tile(symbol_bgr: np.ndarray,
                     binary_symbol: np.ndarray | None,
                     template_img: np.ndarray | None,
                     tile_h: int = 150,
                     tile_w: int = 150) -> np.ndarray:

    def blank_block():
        return np.zeros((tile_h, tile_w // 3, 3), dtype=np.uint8)

    def place_into_block(img, block):
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return block

        # масштабируем по высоте
        scale = tile_h / float(h)
        new_w = max(1, int(w * scale))
        resized = cv2.resize(img, (new_w, tile_h), interpolation=cv2.INTER_NEAREST)

        # если шире блока — уменьшаем до ширины блока
        max_w = block.shape[1]
        if resized.shape[1] > max_w:
            resized = cv2.resize(resized, (max_w, tile_h), interpolation=cv2.INTER_AREA)
            new_w = max_w

        # центрирование по ширине
        x0 = (max_w - new_w) // 2
        block[:, x0:x0 + new_w] = resized
        return block

    # original
    if symbol_bgr is None or symbol_bgr.size == 0:
        orig = blank_block()
    else:
        orig = place_into_block(symbol_bgr, blank_block())

    # binary
    if binary_symbol is None or binary_symbol.size == 0:
        bin_block = blank_block()
    else:
        b3 = cv2.cvtColor(binary_symbol, cv2.COLOR_GRAY2BGR)
        bin_block = place_into_block(b3, blank_block())

    # template
    if template_img is None or template_img.size == 0:
        tmpl_block = blank_block()
    else:
        t3 = cv2.cvtColor(template_img, cv2.COLOR_GRAY2BGR)
        tmpl_block = place_into_block(t3, blank_block())

    # склейка трёх частей
    tile = np.hstack([orig, bin_block, tmpl_block])
    return tile



def build_symbols_collage(tiles: List[np.ndarray],
                          tile_h: int = 150,
                          tile_w: int = 150,
                          cols: int = 4,
                          rows: int = 2) -> np.ndarray:
    """
    Строит коллаж 4×2 из 8 миниатюр.
    """
    total = cols * rows
    # дополнить до нужного количества пустыми
    while len(tiles) < total:
        tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))

    collage_h = rows * tile_h
    collage_w = cols * tile_w
    collage = np.zeros((collage_h, collage_w, 3), dtype=np.uint8)

    for idx in range(total):
        r = idx // cols
        c = idx % cols
        y1 = r * tile_h
        x1 = c * tile_w
        collage[y1:y1 + tile_h, x1:x1 + tile_w] = tiles[idx]

    return collage


# ============================================================
# Распознавание номера
# ============================================================

def recognize_number_from_roi(roi_bgr: np.ndarray):
    """
    Возвращает:
      text: распознанный номер
      tiles_data: список (symbol_bgr, binary_symbol, template_img) для 8 символов
    """
    roi_bgr = adjust_brightness(roi_bgr)
    symbols = split_number_by_image(roi_bgr)

    recognized = ""
    tiles_data: List[Tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = []

    for i, symbol in enumerate(symbols):
        is_digit = i in [1, 2, 3, 6, 7]
        is_region_digit = i in [6, 7]

        best_char, bin_sym, tmpl_img = compare_symbol_with_templates(
            symbol, TEMPLATES_DIR, is_digit, is_region_digit
        )

        recognized += best_char if best_char else "?"
        tiles_data.append((symbol, bin_sym, tmpl_img))

    if len(recognized) == 8:
        text = recognized[:6] + " " + recognized[6:]
    else:
        text = recognized

    return text, tiles_data


# ============================================================
# Обработка изображения целиком
# ============================================================

def process_image(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        print("Не удалось загрузить:", image_path)
        return

    roi, rect, M = find_best_plate_roi(img)

    if roi is None or rect is None:
        print(f"{image_path.name}: номер не найден")
        return

    number_text, tiles_data = recognize_number_from_roi(roi)
    print(f"{image_path.name}: {number_text}")

    # --- рисуем рамку вокруг номера ---
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    boxed = img.copy()
    cv2.drawContours(boxed, [box], 0, (0, 255, 0), 3)

    # --- текст ---
    cv2.putText(
        boxed,
        number_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 0, 255),
        3,
    )

    # --- миниатюра ROI в углу ---
    roi_small = cv2.resize(
        roi,
        (int(roi.shape[1] * 0.7), int(roi.shape[0] * 0.7)),
        interpolation=cv2.INTER_AREA,
    )
    H0, W0 = boxed.shape[:2]
    rh, rw = roi_small.shape[:2]
    if rh + 50 < H0 and rw + 20 < W0:
        boxed[50:50 + rh, W0 - rw - 20:W0 - 20] = roi_small

    # --- строим миниатюры для всех символов ---
    tiles: List[np.ndarray] = []
    for symbol_bgr, bin_sym, tmpl_img in tiles_data:
        tile = make_symbol_tile(symbol_bgr, bin_sym, tmpl_img,
                                tile_h=150, tile_w=150)
        tiles.append(tile)

    collage = build_symbols_collage(tiles, tile_h=150, tile_w=150,
                                    cols=4, rows=2)

    # --- склейка: оригинал сверху, коллаж снизу ---
    h0, w0 = boxed.shape[:2]
    h1, w1 = collage.shape[:2]

    H = h0 + 20 + h1
    W = max(w0, w1)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # исходное изображение по центру сверху
    x0 = (W - w0) // 2
    canvas[0:h0, x0:x0 + w0] = boxed

    # коллаж по центру снизу
    x1 = (W - w1) // 2
    y1 = h0 + 20
    canvas[y1:y1 + h1, x1:x1 + w1] = collage

    out_path = RESULTS_DIR / f"result_{image_path.name}"
    cv2.imwrite(str(out_path), canvas)


# ============================================================
# MAIN
# ============================================================

def main():
    images = sorted(IMAGES_DIR.glob("*.jpg"))
    if not images:
        print("Нет изображений в PR/Images")
        return

    for path in images:
        process_image(path)

    print("Готово. Результаты в PR/Results")


if __name__ == "__main__":
    main()
