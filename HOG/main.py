import numpy as np
import cv2
from scipy.signal import convolve2d as conv2d
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import cv2


Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
file = askopenfilename(initialdir='./',filetypes=[("Images", ["*.jpeg", "*.jpg", "*.png", "*.bmp", "*.dib", "*.tiff", "*.tif"]), ("All files", "*.*")])
# if file != '':
#     img = cv2.imread(file, 0)
#     img = cv2.resize(img, (64, 128))
#     print(img.shape)
#     hog = cv2.HOGDescriptor()
#     hist = hog.compute(img)
#     print(hist.shape)
#     plt.bar(np.arange(len(hist)), hist)
#     plt.show()
#     # cv2.imshow('img', hog_result)
#     # cv2.waitKey(0)
#     # print(hist.shape)

# def hog(img, filtersize = (8, 8)):
#     gradLR = np.abs(conv2d(img, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])))
#     gradUD = np.abs(conv2d(img, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])))



img = cv2.cvtColor(cv2.imread(file),
                   cv2.COLOR_BGR2GRAY)

cell_size = (8, 8)  # h x w in pixels
block_size = (2, 2)  # h x w in cells
nbins = 9  # number of orientation bins

# winSize is the size of the image cropped to an multiple of the cell size
hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                  img.shape[0] // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)

n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
hog_feats = hog.compute(img)\
               .reshape(n_cells[1] - block_size[1] + 1,
                        n_cells[0] - block_size[0] + 1,
                        block_size[0], block_size[1], nbins) \
               .transpose((1, 0, 2, 3, 4))  # index blocks by rows first
# hog_feats now contains the gradient amplitudes for each direction,
# for each cell of its group for each group. Indexing is by rows then columns.

gradients = np.zeros((n_cells[0], n_cells[1], nbins))

# count cells (border cells appear less often across overlapping groups)
cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)

for off_y in range(block_size[0]):
    for off_x in range(block_size[1]):
        gradients[off_y:n_cells[0] - block_size[0] + off_y + 1,
                  off_x:n_cells[1] - block_size[1] + off_x + 1] += \
            hog_feats[:, :, off_y, off_x, :]
        cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
                   off_x:n_cells[1] - block_size[1] + off_x + 1] += 1

# Average gradients
gradients /= cell_count

# Preview
fig, ax = plt.subplots(2)
ax[0].imshow(img, cmap='gray')

bin = 5  # angle is 360 / nbins * direction
ax[1].pcolor(gradients[:, :, bin])
ax[1].invert_yaxis()
ax[1].set_aspect('equal', adjustable='box')
ax[1].colorbar = True
plt.show()
# plt.show(fig[0], fig[1])