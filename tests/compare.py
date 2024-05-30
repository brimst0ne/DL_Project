import cv2
import sys
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compare_pics(image1_path, image2_path):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    m = mse(gray1, gray2)
    s = ssim(gray1, gray2)
    return m, s

def mse(image1, image2):
    diff = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    diff /= float(image1.shape[0] * image1.shape[1])
    return diff

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Неверное количество аргументов!")
        sys.exit(1)
    image_path1 = sys.argv[1]
    image_path2 = sys.argv[2]
    m, s = compare_pics(image_path1, image_path2)
    print("MSE: ", m, "\n")
    print("SSIM: ", s, "\n")