from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def changeShape(arr, rr, rc):
    #合并行后的中间矩阵
    inrm = np.zeros((rr, arr.shape[1]))
    #结果矩阵
    rm = np.zeros((rr, rc))
    #行列的缩小比例
    rratio = int(np.floor(arr.shape[0] / rr))
    cratio = int(np.floor(arr.shape[1] / rc))
    #合并行
    for j in range(rr):
        temp = arr[j * rratio:(j + 1) * rratio, :]
        temp = temp.sum(axis=0)
        for i in range(arr.shape[1]):
            inrm[j, i] = temp[i]
    #合并列
    for j in range(rc):
        temp = inrm[:, j * cratio:(j + 1) * cratio]
        temp = temp.sum(axis=1)
        for i in range(rr):
            if temp[i] > 0:
                rm[i, j] = 1
            else:
                rm[i, j] = 0
    return rm


def showLfigure(filename):
    img = np.array(Image.open(filename).convert('L'))
    print(img)
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            if (img[i, j] <= 75):
                img[i, j] = 0
            else:
                img[i, j] = 1
    plt.figure(filename)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def convertImg2Txt(filename, rrows, rcols):
    img = np.array(Image.open(filename).convert('L'))
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            if (img[i, j] >= 250):
                img[i, j] = 0
            else:
                img[i, j] = 1
    rs = changeShape(img, rrows, rcols)
    np.savetxt(filename + '.txt', rs, fmt="%d", delimiter='')


def convertLsyImg2Txt(filename, rrows, rcols):
    img = np.array(Image.open(filename).convert('L'))
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            if (img[i, j] <= 75):
                img[i, j] = 0
            else:
                img[i, j] = 1
    rs = changeShape(img, rrows, rcols)
    np.savetxt(filename + '.txt', rs, fmt="%d", delimiter='')

# showLfigure('lsy.jpg')
# convertLsyImg2Txt('lsy.jpg',32,64)
