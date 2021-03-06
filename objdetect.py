import cv2
import numpy as np
from matplotlib import pyplot as plt 
from pdf2image import convert_from_path
from PIL import Image
import time 


Image.MAX_IMAGE_PIXELS = 1000000000 


#convert pdf to image

target_img = 'wc.png'
pdf = 'layouts.pdf'
src_img = 'layouts.jpg'
pages = convert_from_path(pdf, 200)
count = 0 

for page in pages:
    t0 = time.time()
    page.save((str(count)+src_img), 'JPEG')
    print("Processing page " + str(count) + " took " + str(time.time() - t0) + " seconds")
    count += 1
'''
#templatematching
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html

for i in range(count):
    img = cv2.imread((str(i) + src_img),0)
    img2 = img.copy()
    template = cv2.imread(target_img,0)
    w, h = template.shape[::-1]

    # All the 6 methods for comparison in a list
    
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    

    methods = ['cv2.TM_SQDIFF']
    
    print("Evaluating method")
    for meth in methods:
        img = img2.copy()
        method = eval(meth)
        # Apply template Matching
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img,top_left, bottom_right, (255,0,0), 4)

        print("Showing matches on image now")
        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)

        plt.show()

'''

#featurematch
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html#matcher
#Brute-Force matching with SIFT Descriptors and Ratio Test

for i in range(count):
    img1 = cv2.imread(target_img,0) # queryImage
    img2 = cv2.imread((str(i)+src_img),0) # trainImage

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)
    print("Showing matches on image now")
    plt.imshow(img3),plt.show()

