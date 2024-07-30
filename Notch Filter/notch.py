import cv2
import numpy as np
# import matplotlib.pyplot as plt
from scipy import fft as f

image = np.double(cv2.imread("clown_noised.jpg", flags=cv2.IMREAD_GRAYSCALE))
fi = f.fftshift(f.fft2(image))
# fi = np.log(fi)
# mask_mat = np.real(fi/np.max(fi) * 255)
# status = cv2.imwrite('mask.png', mask_mat)
# print(status)

m = np.double(cv2.imread("mask3.png", flags=cv2.IMREAD_GRAYSCALE))
m = np.double(m > 0)
g = np.abs(f.ifft2(f.ifftshift(m * fi)))
cv2.imshow("noisy clown", np.uint8(image))
cv2.imshow("clown", np.uint8(g))
cv2.waitKey(0)
