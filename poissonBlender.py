#http://graphics.cs.cmu.edu/courses/15-463/lectures/lecture_09.pdf
#https://www.cs.cornell.edu/courses/cs6640/2012fa/slides/12-GradientDomain.pdf
#https://groups.csail.mit.edu/graphics/classes/CompPhoto06/html/lecturenotes/10_Gradient.pdf
#https://en.wikipedia.org/wiki/Discrete_Poisson_equation

#https://en.wikipedia.org/wiki/Conjugate_gradient_method
#http://www.cs.cmu.edu/~aarti/Class/10725_Fall17/Lecture_Slides/conjugate_direction_methods.pdf

print('Importing libraries...')
import numpy as np
import cv2

import numpy.linalg
from scipy.optimize import nnls
import scipy.sparse.linalg
import conjgrad

import sys

np.set_printoptions(threshold=sys.maxsize)

print('Reading images...')
dest = np.float32(cv2.imread('desert_small.png'))
src = np.float32(cv2.imread('rainbow_small.png'))

x0 = 5
x1 = src.shape[1] - 10

y0 = 10
y1 = src.shape[0] - 50

padx = 20
pady = 0

lenx = x1 - x0
leny = y1 - y0

P = lenx * leny

def v(x, y, i):
    return src[y + 1, x, i] - src[y, x, i]

def u(x, y, i):
    return src[y, x + 1, i] - src[y, x, i]

def f_v(x, y, i):
    return dest[y + 1, x, i] - dest[y, x, i]

def f_u(x, y, i):
    return dest[y, x + 1, i] - dest[y, x, i]

def div_V(x, y, i):
    return u(x + 1, y, i) - u(x, y, i) + v(x, y + 1, i) - v(x, y, i)

def div_f(x, y, i):
    return f_u(x + 1, y, i) - f_u(x, y, i) + f_v(x, y + 1, i) - f_v(x, y, i)

def gradient_V(x, y, i):
    return ( u(x, y, i) , v(x, y, i) )

def gradient_f(x, y, i):
    return ( f_u(x, y, i) , f_v(x, y, i) )

def integrate(dim):
    global dest

    A = np.zeros(P ** 2, dtype=np.float32).reshape((P, P))
    B = np.zeros(P, dtype=np.float32).reshape((P, 1))

    print('Setting up Laplacian...')

    for i, y_ in enumerate(range(y0, y1)):
        for j, x_ in enumerate(range(x0, x1)):
            boundaryX0 = False
            boundaryX1 = False
            boundaryY0 = False
            boundaryY1 = False

            #B[index, :] = div_V(x_, y_, dim) #laplacian(x, y, 0)

            x = x_ + padx
            y = y_ + pady

            index = i * lenx + j

            left_index = i * lenx + j - 1
            right_index = i * lenx + j + 1
            top_index = (i + 1) * lenx + j
            bottom_index = (i - 1) * lenx + j

            A[index, index] = -4

            if x_ == x0:
                B[index] -= dest[pady + i, padx, dim]
                boundaryX0 = True
            else:
                A[index, left_index] = 1
                
            if x_ == x1 - 1:
                B[index] -= dest[pady + i, padx + lenx, dim]
                boundaryX1 = True
            else:
                A[index, right_index] = 1

            if y_ == y0:
                B[index] -= dest[pady, padx + j, dim]
                boundaryY0 = True
            else:
                A[index, bottom_index] = 1

            if y_ == y1 - 1:
                B[index] -= dest[pady + leny, padx + j, dim]
                boundaryY1 = True
            else:
                A[index, top_index] = 1

            if (boundaryX0 or boundaryX1 or boundaryY0 or boundaryY1):
                grad1 = gradient_V(x_, y_, dim)
                grad2 = gradient_f(padx + j, pady + i, dim)

                if np.linalg.norm(grad1) > np.linalg.norm(grad2):
                    B[index, :] += div_V(x_, y_, dim) #laplacian(x, y, 0)
                else:
                    B[index, :] += div_f(padx + j, pady + i, dim) #laplacian(x, y, 0)
            else:
                B[index, :] += div_V(x_, y_, dim)

    print(f'Solving {A.shape[0]}x{A.shape[1]} matrix system...')
    #breakpoint()
    #Naive solution
    #f = nnls(A, np.reshape(B, (B.size,)))

    #Gradient descent
    f = np.zeros(P, dtype=np.float32).reshape((P, 1))

    print('Starting gradient descent...')
    f = scipy.sparse.linalg.cg(A, B, f)

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

final_image = cv2.resize(np.uint8(dest), (5 * dest.shape[1], 5 * dest.shape[0]))

cv2.imwrite('final.png', final_image)

cv2.imshow('Final', final_image)
cv2.waitKey(0)