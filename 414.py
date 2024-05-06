import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 第1步：加载xml文件
face_classifier = cv.CascadeClassifier("../img/haarcascade_frontalface_default.xml")
eye_classifier = cv.CascadeClassifier("../img/haarcascade_eye.xml")

# 第2步：加载图片
lena = cv.imread("../img/lena.jpg")

# 第3步：将图片转成灰色图片
lena_gray = cv.cvtColor(lena, cv.COLOR_BGR2GRAY)

# 第4步：使用api进行人脸识别 参数2：缩放系数  参数3：至少要检测几次才算正确
faces = face_classifier.detectMultiScale(lena_gray, 1.3, 3)
print faces		# 返回值：左上点的x，y，宽w，高h

# 第5步：在人脸上绘制矩形
for x, y, w, h in faces:
    # 从灰色图片中找到人脸
    grayFace = lena_gray[y:y+h, x:x+w]
    colorFace = lena[y:y+h, x:x+w]
    # 在当前人脸上找到眼睛的位置
    eyes = eye_classifier.detectMultiScale(grayFace, 1.2, 5)

	# 在找到人脸上画矩形
    cv.rectangle(lena, (x, y), (x+w, y+h), (0, 0, 255), 2)
	# 在眼睛上绘制矩形
    for eye_x, eye_y, eye_w, eye_h in eyes:
        cv.rectangle(colorFace, (eye_x,  eye_y), (eye_x+eye_w, eye_y+eye_h), (0, 255, 255), 2)

cv.imshow("src", lena)
cv.waitKey()