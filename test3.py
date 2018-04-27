import cv2
import numpy as np

MAX_WIDTH = 500
MAX_HEIGHT = 300

# Input image
img = cv2.imread('1.jpg')

# Resize image
height, width = img.shape[:2]
if MAX_HEIGHT < height or MAX_WIDTH < width:
    scaling_factor = MAX_HEIGHT / float(height)
    if MAX_WIDTH / float(width) < scaling_factor:
        scaling_factor = MAX_WIDTH / float(width)
    img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

# Pre-processing
final_img = img.copy()
copy_img = img.copy()
copy_img = cv2.cvtColor(copy_img, cv2.COLOR_RGB2GRAY)
copy_img = cv2.medianBlur(copy_img, 3)

# Detect Mser Region
creator = cv2.MSER_create()
regions, bboxes = creator.detectRegions(copy_img)

# Make mask contains mser region
mask = np.zeros(copy_img.shape, np.uint8)
for region in regions:
    mask[region[:,1], region[:,0]] = 255
cv2.imshow('mser mask', mask)

# Make canny edegs region
copy_img_2 = img.copy()
gray_img = cv2.cvtColor(copy_img_2, cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(gray_img, (3,3),0)
# _, thresh = cv2.threshold(blur,0 ,255 ,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101,0)
cv2.imshow('thresh',thresh)
egdes = cv2.Canny(thresh, 100, 150)
cv2.imshow('egde',egdes)
kernel = np.ones((5,5),np.uint8)
dilate = cv2.dilate(egdes,kernel,iterations = 1)



_, contours, __ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    cv2.drawContours(dilate, [cnt], -1, 255,-1)
kernel = np.ones((7,7),np.uint8)

erosion = cv2.erode(dilate,kernel,iterations = 1)
cv2.imshow('edges mask',erosion)


# Intersection two mask
intersec_mask = cv2.bitwise_and(erosion, mask)
cv2.imshow('intersec mask', intersec_mask)


# Post processing
post_img = cv2.morphologyEx(intersec_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(9,3)))
post_img = cv2.morphologyEx(post_img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)));
cv2.imshow('final',post_img)

post_mask_img = np.zeros(post_img.shape, np.uint8)

_, contours, __ = cv2.findContours(post_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area
    if solidity > 0.95 or solidity < 0.1:
        continue
    if area < 100:
        continue
    cv2.rectangle(post_mask_img, (x, y), (x + w, y + h), 255, -1)

cv2.imshow('post_mask_img',post_mask_img)
post_mask_img_2 = np.zeros(post_img.shape, np.uint8)
_, contours, __ = cv2.findContours(post_mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h
    area = cv2.contourArea(cnt)
    if aspect_ratio > 1.5 or aspect_ratio < 0.5:
        continue
    if area < 200:
        continue;
    cv2.drawContours(mask, [cnt], -1, 255,-1)

    cv2.rectangle(final_img, (x, y), (x + w, y + h), (0, 255, 0), 1)

cv2.imshow('final_mask',final_img)
# cv2.imwrite('final_mask_2.jpg',mask)

_, contours, __ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# labels = [-1]*len(contours)
# for i in range(len(contours)):
#     if labels[i] == -1:
#         (x, y), (MA, ma), angle = cv2.fitEllipse(contours[i])
#         _, __, ___, h = cv2.boundingRect(contours[i])
#         for j in range(i + 1, len(contours)):
#             if (labels[j] == -1):
#                 (x1, y1), _, __ = cv2.fitEllipse(contours[j])
#                 _, __, ___, h1 = cv2.boundingRect(contours[j])
#                 a = float(x1)/x
#
#                 b = float(y1)/y
#
#                 c = float(h1)/h
#
#                 if ((a < 1.1 and a > 0.8) or (b < 1.1 and b > 0.8)) and (c <1.3 and c>0.8):
#                     labels[j] = i
#                     labels[i] = i
#                     print(a)
#                     print(b)
#                     print(c)
#                     print('dc 1 cap {0} and {1}'.format(i, j))
#
# mask_2 = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
# for i in range(len(contours)):
#     if labels[i] != -1:
# #         cv2.drawContours(mask_2, [contours[i]], -1, 255, -1)
# cv2.imshow('final', img)
# # cv2.imshow('mask_final_final',mask_2)
# # binary image with mask
# final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2GRAY)
# final_img = cv2.GaussianBlur(final_img, (3,3),0)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# final_img = clahe.apply(final_img)
# _,final_img = cv2.threshold(final_img,0 ,255 ,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# final_img = cv2.bitwise_not(final_img)
# # cv2.imshow('cover',final_img)
#
# text_only = cv2.bitwise_and(final_img, final_img, mask=mask)
# text_only = cv2.bitwise_not(text_only)
# text_only = cv2.morphologyEx(text_only, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)));
#
# cv2.imshow('final',text_only)
# cv2.imwrite('text_only.jpg',text_only)
cv2.waitKey(0)



