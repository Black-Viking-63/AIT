import numpy as np
import numba
from numba import cuda
import math
from time import time
import PIL
from PIL import Image

def salt_and_pepper_add(image, prob):
    rnd = np.random.rand(image.shape[0], image.shape[1])
    noisy = image.copy()
    noisy[rnd < prob] = 0
    noisy[rnd > 1 - prob] = 255
    return noisy

def median_filter(a):
    b = a.copy()
    start = time()
    for i in range(2, len(a)-1):
        for j in range(2, len(a[i])-1):
            t=[0, 0, 0, 0, 0, 0, 0, 0, 0]
            t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8] = a[i-1][j-1], a[i-1][j], a[i-1][j+1], a[i][j-1], a[i][j], a[i][j+1], a[i+1][j-1], a[i+1][j], a[i+1][j+1]
            for k in range(8):
                for l in range(8-k):
                    if t[l]>t[l+1]:
                        t[l], t[l+1] = t[l+1], t[l]
            b[i][j]=t[(int)(len(t)/2)]
    return b, time()-start

def experiment(img_name, need_draw):
    im=(Image.open(img_name)).convert('L')
    img = np.array(im)

    img = salt_and_pepper_add(img, 0.09)
    img3 = Image.fromarray(np.uint8(img))
    img3.save('/usr/app/src/SAP.jpg')

    img1, ctime = median_filter(img)
    img4 = Image.fromarray(np.uint8(img1))
    img4.save('/usr/app/src/CPU.jpg')

    n=len(img)*len(img[0])
    print('Количество элементов =', n)
    print('Время работы алгоритма на CPU =', ctime)

print('Начали обработку!')
experiment('/usr/app/src/price.jpg', True)
print('Закончили обработку!')


