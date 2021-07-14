# vehicles_detection_and-Counting


Vehicle Detection  and Counting Using OpenCV Python

In today's episode, we will see how Vehicle  Detection can be done using Python OpenCV  via an Image file, webcam or video file.

First of all you need to install OpenCV . We will do this tutorial using the completed Python programming language so let's get started.






OpenCV Python

 OpenCV
is an image processing library. It is designed to solve computer vision problems. OpenCV is a C/C++ library that is extended in Python.

To install  OpenCV, Here



NumPy
The NumPy library is used to support multi-dimensional arrays, matrices, etc. in the Python programming language. It is an open-source numerical Python library.

Numpy provides:
tools for integrating C/C++ and Fortran program
sophisticated functions
a powerful N-dimensional array item.
useful linear algebra, Fourier transform, and random number capabilities
and more.
    
    To install Numpy, visit here.

Import CV2 and numpy

Now create a file like vehicleDetection.py

 
import cv2
import numpy as np
 
 
cv2.VideoCapture(​video_path​)


If use a video file
 
cap=cv2.VideoCapture('/home/python/OpenCV/vehiclesDetection/vehicles.mp4')
 

If use  webcam,then
 
cap = cv2.VideoCapture(0) #depends on your system 0, -1 or 1
 



VideoCapture.isOpened() function
Returns true if video capturing has been initialized already and  If the previous call to VideoCapture constructor or VideoCapture::open() succeeded, the method returns true.

VideoCapture.read() function
read([, image])   image the video frame is returned here. If no frames has been grabbed the image will be empty.

 
if cap.isOpened():
   ret, frame1 = cap.read()
else:
   ret = False
ret, frame1 = cap.read()
ret, frame2 = cap.read()
 



Define some variables:
 
min_contour_width = 40  
min_contour_height = 40  
offset = 10  
line_height = 550  
matches = []
cars = 0
 

Create a function

 
def get_centrolid(x, y, w, h):
   x1 = int(w / 2)
   y1 = int(h / 2)
 
   cx = x + x1
   cy = y + y1
   return cx, cy
 

VideoCapture.set() function
set(propId, value)  sets a property in the VideoCapture. propId Property identifier from cv::VideoCaptureProperties (eg. cv::CAP_PROP_POS_MSEC, cv::CAP_PROP_POS_FRAMES, ...) .return true. if the property is supported by the backend used by the VideoCapture instance.

 
cap.set(3, 1920)
cap.set(4, 1080)
 
 


cv2.absdiff() function

Returns values(MSubArray | D | mvoid | complex64 | float16 | uint32 | MaskedConstant | int16 | ndarray | int64 | int32 | memmap | number | bool_ | timedelta64 | float128 | int8 | matrix | uint8 | integer | datetime64 | MaskedArray | recarray | uint64 | float64)

absdiff(​src1, src2​)


 
d =cv2.absdiff(frame1, frame2)
  


cv2.cvtColor(src, code[, dst[, dstCn]]) 

Converts an image from one color space to another. The function converts an input image from one color space to another. In case of a transformation . to-from RGB color space, the order of the channels should be specified explicitly (RGB or BGR). Note . that the default color format in OpenCV is often referred to as RGB but it is actually BGR (the . bytes are reversed). 

   
grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
 

cv2.dilate() function
This function accepts dilate(src, kernel[, dst[, anchor[, iterations[, borderType[ , borderValue]]]]])
Dilates an image by using a specific structuring element. The function dilates the source image using the specified structuring element that determines the shape of a pixel neighborhood .

 
dilated = cv2.dilate(th, np.ones((3, 3)))
 

 cv2.threshold() function
threshold(src, thresh, maxval, type[, dst]) Applies a fixed-level threshold to each array element. The function applies fixed-level thresholding to a multiple-channel array. The function is typically used to get a bi-level (binary) image out of a grayscale image  or for removing a noise, that is, filtering out pixels with too small or too large values. There are several types of thresholding supported by the function. They are determined by type parameter.

 
   ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
 

cv2.GaussianBlur() function

GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) 
 Blurs an image using a Gaussian filter.The function convolves the source image with the specified Gaussian kernel. In-place filtering is supported.
src input image; the image can have any number of channels, which are processed.

 
   blur = cv2.GaussianBlur(grey, (5, 5), 0)
 
 
 

cv2.getStructuringElement() function
getStructuringElement(shape, ksize[, anchor])Returns a structuring element of the specified size and shape for morphological operations. The function constructs and returns the structuring element that can be further passed to #erode, #dilate or #morphologyEx. But you can also construct an arbitrary binary mask yourself and use it as the structuring element. shape Element shape that could be one of #MorphShapes

 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
 
 

cv2.morphologyEx() function
morphologyEx(src, op, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) Performs advanced morphological transformations.
The function cv2.morphologyEx can perform advanced morphological transformations using erosion and dilation as basic operations.
Any of the operations can be done in-place. In case of multi-channel images, each channel is processed independently.

   
closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
 

cv2.findContours() function
findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> contours, hierarchy. It finds contours in a binary image.
The function retrieves contours from the binary image using the algorithm. The contours are a useful tool for shape analysis and object detection and recognition.

CHAIN_APPROX_SIMPLE([x])
Convert a number or string to an integer, or return 0 if no arguments are given. If x is a number, return x.__int__(). For floating point numbers, this truncates towards zero.

  
 contours, h = cv2.findContours(
       closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 


cv2.boundingRect(array) function

Calculates the up-right bounding rectangle of a point set or non-zero pixels of gray-scale image.The function calculates and returns the minimal up-right bounding rectangle for the specified point set or non-zero pixels of gray-scale image.



cv2.putText(img, text, org, fontFace, fontScale, color[ , thickness[ , lineType[, bottomLeftOrigin]]]) 

img input Draws a text string,
text Text string to be drawn.
org Bottom-left corner of the text string in the image. 
fontFace Font type, 
fontScale Font scale factor that is multiplied by the font-specific base size.
color Text color.
thickness Thickness of the lines used to draw a text. 
lineType Line type. 
bottomLeftOrigin When true, the image data origin is at the bottom-left corner. Otherwise, it is at the top-left corner.






 
 
while ret:
   d = cv2.absdiff(frame1, frame2)
   grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
 
   blur = cv2.GaussianBlur(grey, (5, 5), 0)
 
   ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
   dilated = cv2.dilate(th, np.ones((3, 3)))
   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
 
  
   closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
   contours, h = cv2.findContours(
       closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   for(i, c) in enumerate(contours):
       (x, y, w, h) = cv2.boundingRect(c)
       contour_valid = (w >= min_contour_width) and (
           h >= min_contour_height)
 
       if not contour_valid:
           continue
       cv2.rectangle(frame1, (x-10, y-10), (x+w+10, y+h+10), (255, 0, 0), 2)
 
       cv2.line(frame1, (0, line_height), (1200, line_height), (0, 255, 0), 2)
       centrolid = get_centrolid(x, y, w, h)
       matches.append(centrolid)
       cv2.circle(frame1, centrolid, 5, (0, 255, 0), -1)
       cx, cy = get_centrolid(x, y, w, h)
       for (x, y) in matches:
           if y < (line_height+offset) and y > (line_height-offset):
               cars = cars+1
               matches.remove((x, y))
               
 
   cv2.putText(frame1, "Total Cars Detected: " + str(cars), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
               (0, 170, 0), 2)
 
 
 






 Show image cv2.imshow() function.

  
 cv2.imshow("Vehicle Detection", frame1)
  
   if cv2.waitKey(1) == 27:
       break
   frame1 = frame2
   ret, frame2 = cap.read()
 
 






