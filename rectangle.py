# import required libraries
import cv2
import numpy as np
drawing = False
ix,iy = -1,-1

x0 = -1
x1 = -1
y0 = -1
y1 = -1

# define mouse callback function to draw circle
def draw_rectangle(event, x, y, flags, param):
   global ix, iy, drawing, img, x0, x1, y0, y1
   if event == cv2.EVENT_LBUTTONDOWN:
      drawing = True
      ix = x
      iy = y
   elif event == cv2.EVENT_LBUTTONUP:
      drawing = False
      #v2.rectangle(img, (ix, iy),(x, y),(0, 255, 255),-1)
      sub_img = img[iy:y, ix:x]
      white_rect = np.uint8(np.ones(sub_img.shape, dtype=np.uint8) * (0,255,255))
      
      res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
      img[iy:y, ix:x] = res

      x0 = ix
      x1 = x

      y0 = iy
      y1 = y

def selector(img):
   # Create a window and bind the function to window
    cv2.namedWindow("Rectangle Window")

    # Connect the mouse button to our callback function
    cv2.setMouseCallback("Rectangle Window", draw_rectangle)

    # display the window
    while True:
        cv2.imshow("Rectangle Window", img)
        if cv2.waitKey(10) == 27:
            cv2.destroyAllWindows()
            break
    
    return (x0, x1, y0, y1)