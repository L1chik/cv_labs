import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip
from modelhub_client import (ModelHub,
                             models_example)


os.makedirs("result", exist_ok=True)

model_hub = ModelHub(models=models_example,
                     local_storage=os.path.join(os.getcwd(), "./data"))
model_hub.download_model_by_name("numberplate_options")

number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading",
                                              image_loader="opencv")
images_arr = ['./1.jpg', './2.jpg', './3.jpg','./4.jpg','./5.jpg','./6.jpg','./7.jpg','./8.jpg']

(images, images_bboxs,
 images_points, images_zones, region_ids,
 region_names, count_lines,
 confidences, texts) = unzip(number_plate_detection_and_reading(images_arr))

font = cv.FONT_HERSHEY_SIMPLEX

for img, zones, all_points, pp in zip(images, images_zones, images_points, texts):
    img2 = img.copy()

    for points in all_points:
        pts = points.astype('int32')
        cv.polylines(img2, [pts], True, (0, 0, 255), 3) #отображение bbox


    for zone in zones:
        zone2 = zone.copy()
        cv.putText(img2, ''.join(pp), (20, 400), font, 1.3, (0, 255, 0), 3, cv.LINE_AA)

        h0, w0 = img2.shape[:2]
        h1, w1 = zone2.shape[:2]

        h = h0 + 20 + h1
        w = max(w0, w1)

        canvas = np.zeros((w, w, 3), dtype=np.uint8)

        # исходное изображение по центру сверху
        x0 = (w - w0) // 2
        canvas[0:h0, x0:x0 + w0] = img2

        x1 = (w - w1) // 2
        y1 = h0 + 20
        canvas[y1:y1 + h1, x1:x1 + w1] = zone2

        cv.imwrite(f"result/output_{pp}.jpg", canvas)



    plt.imshow(img2)
    plt.show()

