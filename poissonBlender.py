#https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf
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

import sys

np.set_printoptions(threshold=sys.maxsize)

print('Reading images...')
dest = np.float32(cv2.imread('nyc.jpg'))
src = np.float32(cv2.imread('trex.jpg'))

x0 = 20
x1 = 180 #src.shape[1] - 10

y0 = 10
y1 = 215 # src.shape[0] - 10

padx = 100
pady = 50

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

            left_index = i * lenx + j - 1
            right_index = i * lenx + j + 1
            top_index = (i + 1) * lenx + j
            bottom_index = (i - 1) * lenx + j

            A[index, index] = -4
            B[index, :] = div_V(x_, y_, dim)

            #neighbours = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        
            #for n in neighbours:
            #    gradient_source = src[y_ + n[0], x_ + n[1], dim] - src[y_, x_, dim]
            #    gradient_target = dest[pady + i + n[0], padx + j + n[1], dim] - dest[pady + i, padx + j, dim]

            #    if abs(gradient_source) > abs(gradient_target):
            #        B[i] += gradient_source
            #    else:
            #        B[i] += gradient_target

            #magnitude_grad_g = np.linalg.norm(gradient(src, x_, y_, dim))
            #magnitude_grad_f = np.linalg.norm(gradient(dest, padx + j, pady + i, dim))
            #print(magnitude_grad_g, magnitude_grad_f)

            #if magnitude_grad_g > magnitude_grad_f:
            #    B[index, :] += divergenceOfGradient(src, x_, y_, dim)
            #else:
            #    B[index, :] += divergenceOfGradient(dest, padx + j, pady + i, dim)

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

    print(f'Solving {A.shape[0]}x{A.shape[1]} matrix system...')
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

        dest[y_coordinate, x_coordinate, dim] = np.float32(f[0][i])

integrate(0)
integrate(1)
integrate(2)

dest = np.uint8(np.clip(dest, 0, 250))
#final_image = cv2.resize(dest, (5 * dest.shape[1], 5 * dest.shape[0]))

cv2.imwrite('final.png', dest)

cv2.imshow('Final', dest)
cv2.waitKey(0)