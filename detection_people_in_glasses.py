"""
Created on Thu Aug 16 22:20:37 2018
@author: James Wu
"""

import dlib
import cv2
import numpy as np

from settings import DLIB_PREDICTOR_PATH

#==============================================================================
#   1.landmarks格式转换函数 
#       输入：dlib格式的landmarks
#       输出：numpy格式的landmarks
#==============================================================================   
def landmarks_to_np(landmarks, dtype="int"):
    # 获取landmarks的数量
    num = landmarks.num_parts
    
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((num, 2), dtype=dtype)
    
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

#==============================================================================
#   2.绘制回归线 & 找瞳孔函数
#       输入：图片 & numpy格式的landmarks
#       输出：左瞳孔坐标 & 右瞳孔坐标
#==============================================================================   
def get_centers(img, landmarks):
    # 线性回归
    EYE_LEFT_OUTTER = landmarks[2]
    EYE_LEFT_INNER = landmarks[3]
    EYE_RIGHT_OUTTER = landmarks[0]
    EYE_RIGHT_INNER = landmarks[1]

    x = ((landmarks[0:4]).T)[0]
    y = ((landmarks[0:4]).T)[1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    
    x_left = (EYE_LEFT_OUTTER[0]+EYE_LEFT_INNER[0])/2
    x_right = (EYE_RIGHT_OUTTER[0]+EYE_RIGHT_INNER[0])/2
    LEFT_EYE_CENTER =  np.array([np.int32(x_left), np.int32(x_left*k+b)])
    RIGHT_EYE_CENTER =  np.array([np.int32(x_right), np.int32(x_right*k+b)])
    
    return LEFT_EYE_CENTER, RIGHT_EYE_CENTER

#==============================================================================
#   3.人脸对齐函数
#       输入：图片 & 左瞳孔坐标 & 右瞳孔坐标
#       输出：对齐后的人脸图片
#============================================================================== 
def get_aligned_face(img, left, right):
    desired_w = 256
    desired_h = 256
    desired_dist = desired_w * 0.5
    
    eyescenter = ((left[0]+right[0])*0.5 , (left[1]+right[1])*0.5)# 眉心
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    dist = np.sqrt(dx*dx + dy*dy)# 瞳距
    scale = desired_dist / dist # 缩放比例
    angle = np.degrees(np.arctan2(dy,dx)) # 旋转角度
    M = cv2.getRotationMatrix2D(eyescenter,angle,scale)# 计算旋转矩阵

    # update the translation component of the matrix
    tX = desired_w * 0.5
    tY = desired_h * 0.5
    M[0, 2] += (tX - eyescenter[0])
    M[1, 2] += (tY - eyescenter[1])

    aligned_face = cv2.warpAffine(img,M,(desired_w,desired_h))
    
    return aligned_face

#==============================================================================
#   4.是否戴眼镜判别函数
#       输入：对齐后的人脸图片
#       输出：判别值(True/False)
#============================================================================== 
def judge_eyeglass(img):
    img = cv2.GaussianBlur(img, (11,11), 0) #高斯模糊

    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0 ,1 , ksize=-1) #y方向sobel边缘检测
    sobel_y = cv2.convertScaleAbs(sobel_y) #转换回uint8类型

    edgeness = sobel_y #边缘强度矩阵
    
    #Otsu二值化
    retVal,thresh = cv2.threshold(edgeness,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    #计算特征长度
    d = len(thresh) * 0.5
    x = np.int32(d * 6/7)
    y = np.int32(d * 3/4)
    w = np.int32(d * 2/7)
    h = np.int32(d * 2/4)

    x_2_1 = np.int32(d * 1/4)
    x_2_2 = np.int32(d * 5/4)
    w_2 = np.int32(d * 1/2)
    y_2 = np.int32(d * 8/7)
    h_2 = np.int32(d * 1/2)
    
    roi_1 = thresh[y:y+h, x:x+w] #提取ROI
    roi_2_1 = thresh[y_2:y_2+h_2, x_2_1:x_2_1+w_2]
    roi_2_2 = thresh[y_2:y_2+h_2, x_2_2:x_2_2+w_2]
    roi_2 = np.hstack([roi_2_1,roi_2_2])
    
    measure_1 = sum(sum(roi_1/255)) / (np.shape(roi_1)[0] * np.shape(roi_1)[1])#计算评价值
    measure_2 = sum(sum(roi_2/255)) / (np.shape(roi_2)[0] * np.shape(roi_2)[1])#计算评价值
    measure = measure_1*0.3 + measure_2*0.7
    
    #根据评价值和阈值的关系确定判别值
    if measure > 0.15:#阈值可调，经测试在0.15左右
        judge = True
    else:
        judge = False
    return judge

#==============================================================================
#   **************************主函数入口***********************************
#==============================================================================

detector = dlib.get_frontal_face_detector()#人脸检测器detector
predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)#人脸关键点检测器predictor


def find(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 1)
    results = []

    for rect in rects:
        x_face = rect.left()
        y_face = rect.top()
        w_face = rect.right() - x_face
        h_face = rect.bottom() - y_face
        
        landmarks = predictor(gray, rect)
        landmarks = landmarks_to_np(landmarks)

        LEFT_EYE_CENTER, RIGHT_EYE_CENTER = get_centers(img, landmarks)
        
        aligned_face = get_aligned_face(gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)
        judge = judge_eyeglass(aligned_face)

        results.append((w_face, h_face, judge))
    return results