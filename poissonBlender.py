#http://graphics.cs.cmu.edu/courses/15-463/lectures/lecture_09.pdf
#https://www.cs.cornell.edu/courses/cs6640/2012fa/slides/12-GradientDomain.pdf
#https://en.wikipedia.org/wiki/Discrete_Poisson_equation

print('Importing libraries...')
import numpy as np
import cv2

import numpy.linalg
from scipy.optimize import nnls

import sys

np.set_printoptions(threshold=sys.maxsize)


print('Reading images...')
src = np.float32(cv2.imread('trump.jpeg'))
dest = np.float32(cv2.imread('monalisa.jpeg'))

x0 = 56
x1 = 93

y0 = 31
y1 = 70

padx = 59
pady = 48

lenx = x1 - x0
leny = y1 - y0

P = lenx * leny

def v(x, y, i):
    return src[y + 1, x, i] - src[y, x, i]

def u(x, y, i):
    return src[y, x + 1, i] - src[y, x, i]

def div_V(x, y, i):
    return u(x + 1, y, i) - u(x, y, i) + v(x, y + 1, i) - v(x, y, i)

def integrate(dim):
    global dest

    A = np.zeros(P ** 2, dtype=np.float32).reshape((P, P))
    B = np.zeros(P, dtype=np.float32).reshape((P, 1))

    print('Setting up Laplacian...')

    for i, y_ in enumerate(range(y0, y1)):
        for j, x_ in enumerate(range(x0, x1)):
            x = x_ + padx
            y = y_ + pady

            index = i * lenx + j

            B[index, :] = div_V(x_, y_, dim) #laplacian(x, y, 0)

            left_index = i * lenx + j - 1
            right_index = i * lenx + j + 1
            top_index = (i + 1) * lenx + j
            bottom_index = (i - 1) * lenx + j

            A[index, index] = -4

            if x_ == x0:
                B[index] -= dest[pady + i, padx, dim]
            else:
                A[index, left_index] = 1
                
            if x_ == x1 - 1:
                B[index] -= dest[pady + i, padx + lenx, dim]
            else:
                A[index, right_index] = 1

            if y_ == y0:
                B[index] -= dest[pady, padx + j, dim]
            else:
                A[index, bottom_index] = 1

            if y_ == y1 - 1:
                B[index] -= dest[pady + leny, padx + j, dim]
            else:
                A[index, top_index] = 1

    print('Solving...')
    f = nnls(A, np.reshape(B, (B.size,)))

    print('Formatting...')
    for i in range(len(f[0])):
        x_coordinate = i % lenx + padx
        y_coordinate = i // lenx + pady

        if f[0][i] < 255:
            dest[y_coordinate, x_coordinate, dim] = np.float32(f[0][i])
        else:
            dest[y_coordinate, x_coordinate, dim] = 250

integrate(0)
integrate(1)
integrate(2)

cv2.imwrite('final.png', np.uint8(dest))
quit()

cv2.imshow('Final', dest / 255)
cv2.waitKey(0)