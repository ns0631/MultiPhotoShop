import numpy as np
import cv2, sys
from matplotlib import pyplot as plt

from scipy.sparse import lil_matrix
from scipy.sparse import linalg

# return all points' coordiantes contained in the selected region
def get_mask_points(img, polygon_coordinates):
    mask_image = np.zeros(img.shape, np.float64)  
    mask_image = cv2.drawContours(mask_image, [polygon_coordinates], -1, color=1.0, thickness=-1)
    mask_points = np.array(np.where(mask_image == 1.0)).transpose()
    
    return mask_points

def find_boundary(img, mask_points):
    mask_points = mask_points.tolist()
    h, w = img.shape
    boundary = []

    for point in mask_points:
        x, y = point[0], point[1]
        neighbours = [[x-1, y], [x+1, y], [x, y-1], [x, y+1]]
        for n in neighbours:
            # check if the neighbour is in the image scale
            if n[0] >= 0 and n[0] < h and n[1] >= 0 and n[1] < w:
                # check if the neighbours is (not in the selected region) and (not yet in the boundary list)
                if (n not in mask_points) and (n not in boundary):
                    boundary.append(n)

    return np.array(boundary)

def build_matrix_A(mask_points):
    N = mask_points.shape[0]
    mask_points = mask_points.tolist()
    A = lil_matrix((N, N))
    
    for i in range(N):

        # put 4 on diagonal
        A[i,i] = -4

        # get the four neighbours of current variable,
        # if the neighbour is in the mask, add -1 to its index
        x, y = mask_points[i][0], mask_points[i][1]
        neighbours = [[x-1, y], [x+1, y], [x, y-1], [x, y+1]]
        for n in neighbours:
            if n in mask_points:
                index = mask_points.index(n)
                A[i, index] = 1
    
    return A.tocsr()

# solve Ax = b

def solve_grey_scale(img, mask_points):
    boundary =  find_boundary(img, mask_points)

    A = build_matrix_A(mask_points)
    b = build_matrix_b(img, mask_points, boundary)
    x = np.zeros(b.size, dtype=b.dtype)
    x = linalg.cg(A, b, x)

    return x

# rebuild image with the Poisson Equation solution

def rebuild_grey(img, mask_points, x):
    img_rebuild = img.copy()

    for i, point in enumerate(mask_points):
        img_rebuild[point[0], point[1]] = x[i]
    
    return img_rebuild

# load source and destination image
source = cv2.imread(sys.argv[1])
source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
source = source/255.

target = cv2.imread(sys.argv[2])
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
target = target/255.

source_polygon = np.array( [ [20, 10], [180, 10], [180, 215], [20, 215] ] )

# define a same polygon in the destination image with a transformation
x_translate, y_translate = 150, 100
target_polygon = np.copy(source_polygon)
target_polygon[:,0] += x_translate
target_polygon[:,1] += y_translate

# get list of points in the selected region from target and source
mask_target = get_mask_points(target_gray, target_polygon)
mask_source = get_mask_points(source_gray, source_polygon)

# Cloning
cloning = target_gray.copy()
for i, point in enumerate(mask_target):
    cloning[point[0], point[1]] = source_gray[mask_source[i][0], mask_source[i][1]]

def Laplacian_source(img, mask_points):
    N = mask_points.shape[0]
    Laplacian = np.zeros(N)

    for i in range(N):
        x, y = mask_points[i][0], mask_points[i][1]
        Laplacian[i] = -4 * img[x, y] + img[x-1, y] + img[x+1, y] + img[x, y-1] + img[x, y+1]

    return Laplacian

def build_matrix_b(img, mask_points, boundary, Laplacian):
    N = mask_points.shape[0]
    mask_points = mask_points.tolist()
    boundary = boundary.tolist()
    b = np.zeros((N, 1))

    for i in range(N):
        
        b[i] += Laplacian[i]

        # get neighbours of the current variable
        # check if they are in the boundary
        x, y = mask_points[i][0], mask_points[i][1]
        neighbours = [[x-1, y], [x+1, y], [x, y-1], [x, y+1]]
        for n in neighbours:
            if n in boundary:
                b[i] -= img[n[0], n[1]]

    return b

def Laplacian_mixing_gradients(source, target, mask_source, mask_target):

    N = mask_source.shape[0]
    Laplacian = np.zeros(N)

    for i in range(N):
        
        # compare the gradient field of source and target
        # keep the gradient with higher absolute value

        x_source, y_source = mask_source[i][0], mask_source[i][1]
        x_target, y_target = mask_target[i][0], mask_target[i][1]
        neighbours = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        
        for n in neighbours:
            gradient_source = source[x_source + n[0], y_source + n[1]] - source[x_source, y_source]
            gradient_target = target[x_target + n[0], y_target + n[1]] - target[x_target, y_target]

            if abs(gradient_source) > abs(gradient_target):
                Laplacian[i] += gradient_source
            else:
                Laplacian[i] += gradient_target

    return Laplacian

def solve_mixing_gradiants(source, target, mask_source, mask_target):
    boundary = find_boundary(target, mask_target)
    mixing_Laplacian = Laplacian_mixing_gradients(source, target, mask_source, mask_target)
    
    A = build_matrix_A(mask_target)
    b = build_matrix_b(target, mask_target, boundary, mixing_Laplacian)
    x = np.zeros(b.size, dtype=b.dtype)
    x = linalg.cg(A, b, x)

    return x[0]

source_rgb = cv2.cvtColor(np.uint8(255 * source), cv2.COLOR_BGR2RGB)
source_rgb = source_rgb/255.

target_rgb = cv2.cvtColor(np.uint8(255 * target), cv2.COLOR_BGR2RGB)
target_rgb = target_rgb/255.

# plot images
f, (ax1, ax2) = plt.subplots(1, 2) 
ax1.title.set_text("destination")
ax1.imshow(target_rgb)
ax2.title.set_text("source")
ax2.imshow(source_rgb)
plt.show()

def rebuild_rgb(source_rgb, target_rgb, mask_source, mask_target):
    
    # build new image with 3 channels
    rebuild_rgb = np.zeros(target_rgb.shape)

    # solve the equation for rgb channels separately
    for i in range(3):
        source_per_channel = source_rgb[:,:,i]
        target_per_channel = target_rgb[:,:,i]
        x_per_channel = solve_mixing_gradiants(source_per_channel, target_per_channel, mask_source, mask_target)
        x_per_channel = np.clip(x_per_channel, 0., 1.)
        rebuild_rgb[:,:,i] = rebuild_grey(target_per_channel, mask_target, x_per_channel)

    return rebuild_rgb

rebuild_rgb_ = rebuild_rgb(source_rgb, target_rgb, mask_source, mask_target)

plt.figure(figsize = (5,5))
plt.imshow(rebuild_rgb_)
plt.show()