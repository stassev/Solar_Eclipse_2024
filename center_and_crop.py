#Copyright: Svetlin Tassev.
#Released under GPLv3 license.

#Re-center and crop images, so that the eclipsed Sun is in
#the center of the image. Use Hough transform to
#find the center of the eclipsed Sun to subpixel precision.
#The algorithm then upscales image, then recenter and crop, then downscale.


import cv2
import numpy as np
import os

dir = '/home/user/Pictures/Eclipse2024/'
scale_factor = 3.0
Rsun=510 # approximate radius of the Sun in pixels
S = round(10 * Rsun) # The desired final size of the images in pixels. It is 10 times the radius of the eclipsed Sun.
#Black borders are added if canvas of resulting image extends beyond the original image.
if (S % 2 == 0):
    S += 1
new_width = new_height = S * scale_factor
k=0
for filename in os.listdir(dir):
    if filename.endswith('.exr') and filename.startswith('DSC'):
        k+=1
        # Load the image as a 32-bit depth NumPy array
        img = cv2.imread(os.path.join(dir, filename), flags=(cv2.IMREAD_COLOR|cv2.IMREAD_ANYDEPTH)).astype(np.float32)
        
        # Detect circles using Hough Transform
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray=((gray)/np.max(gray)*255).astype(np.uint8)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 1000, param1=50, param2=30, minRadius=0.95*Rsun, maxRadius=1.1*Rsun)
        
        # Use the first found circle. Given the min/maxRadius, this should correspond to the Moon/Sun.
        for circle in circles[0]:
            # Resize the original image to 3 times the size
            resized_img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            
            # Create a new canvas with a black background
            canvas = np.zeros((int(new_height), int(new_width), 3), dtype=np.float32)
            
            # Calculate the top-left corner of the original image on the new canvas
            x = int(-scale_factor * circle[0] + new_width / 2 + 0.5)
            y = int(-scale_factor * circle[1] + new_height / 2 + 0.5)
            
            # Composite the resized image onto the new canvas
            # Determine the region of the canvas that the resized image will be placed in
            canvas_x1 = max(0, x)
            canvas_y1 = max(0, y)
            canvas_x2 = min(int(new_width), canvas_x1 + resized_img.shape[1])
            canvas_y2 = min(int(new_height), canvas_y1 + resized_img.shape[0])
            
            # Determine the region of the resized image that will be placed on the canvas
            img_x1 = max(0, -x)
            img_y1 = max(0, -y)
            img_x2 = min(resized_img.shape[1],img_x1 + canvas_x2 - canvas_x1)
            img_y2 = min(resized_img.shape[0], img_y1 + canvas_y2 - canvas_y1)
            
            canvas_x2=min(int(new_width), img_x2-img_x1+canvas_x1)
            canvas_y2=min(int(new_height), img_y2-img_y1+canvas_y1)
            
            # Composite the resized image onto the new canvas
            canvas[canvas_y1:canvas_y2, canvas_x1:canvas_x2] = resized_img[img_y1:img_y2, img_x1:img_x2]
        
            # Resize the cropped image back to the original resolution
            cropped_img = cv2.resize(canvas, (int(new_width / scale_factor + 0.5), int(new_height / scale_factor + 0.5)), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            
            # Save the cropped image as a 32-bit TIF file
            cv2.imwrite(dir+f'cropped_{os.path.splitext(filename)[0]}.exr', cropped_img)
            
            print(f"Center coordinates: ({circle[0]}, {circle[1]})")
            print(f"Radius: {circle[2]}")
