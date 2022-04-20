# Họ tên: Lê Thị Thanh Thanh
# MSSV: 19520954

import cv2 as cv

# Bước 1: Đọc ảnh feature map

featuremaps = cv.imread('FeatureMap.png')
fm_gray = cv.cvtColor(featuremaps, cv.COLOR_BGR2GRAY)

img_input = cv.imread('Input.png')

w_input = img_input.shape[0]
h_input = img_input.shape[1]
w_fm = fm_gray.shape[0]
h_fm = fm_gray.shape[1]

# Bước 2: Xác định các thành phần liên thông của vùng sáng trên feature map (sử dụng thư viện của OpenCV)
_, threshold = cv.threshold(fm_gray, 100, 255, 0)
cv.imshow('Threshold', threshold)
cv.waitKey(0)
cv.destroyAllWindows()

# Bước 3: Xác định bounding box của các thành phần liên thông (sử dụng thư viện của OpenCV).

contours = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
print(len(cnt))

cv.drawContours(fm_gray, cnt, -1, (255,0,0), 2)

# Bước 4: Nội suy ra bounding box của gương mặt trên ảnh gốc

for c in cnt:
    x, y, w, h = cv.boundingRect(c)
    _x = int(w_input * x/w_fm)
    _y = int(h_input * y/h_fm)
    _h = int(h_input * h/h_fm)
    _w = int(w_input * w/w_fm)
        
    print(_x, _y, _w, _h)
    cv.rectangle(featuremaps, (x,y), (x + w, y + h), (0,255,0), 1)
    cv.rectangle(img_input, (_x,_y), (_x + _w, _y + _h), (0,255,0), 2)

cv.imshow('bounding box in featuremaps', featuremaps)
cv.imwrite('bbInfeaturemaps.png', featuremaps)
cv.waitKey(0)

cv.imshow('bounding box in Input', img_input)
cv.imwrite('bbInInput.png', img_input)
cv.waitKey(0)

cv.destroyAllWindows()

