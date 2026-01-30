import cv2
import numpy as np

def analyze_image(img, cfg):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    contrast = gray.std()
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    is_grayscale = (
        np.std(img[:, :, 0] - img[:, :, 1]) < cfg["grayscale_tol"] and
        np.std(img[:, :, 1] - img[:, :, 2]) < cfg["grayscale_tol"]
    )

    return {
        "blur": blur_score < cfg["blur_th"],
        "low_brightness": brightness < cfg["low_brightness"],
        "high_brightness": brightness > cfg["high_brightness"],
        "grayscale": is_grayscale,
        "brightness": brightness,
        "contrast": contrast,
        "blur_score": blur_score,
    }
