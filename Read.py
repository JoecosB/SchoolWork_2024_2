import cv2
import numpy as np

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    squares = []
    img = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin = cv2.Canny(gray, 30, 100, apertureSize=3)    
    contours, _hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    index = 0
    # 轮廓遍历
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True) #计算轮廓周长
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True) #多边形逼近
        # 条件判断逼近边的数量是否为4，轮廓面积是否大于1000，检测轮廓是否为凸的
        if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
            M = cv2.moments(cnt) #计算轮廓的矩
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])#轮廓重心
            
            cnt = cnt.reshape(-1, 2)
            if True:
                index = index + 1
                squares.append(cnt)
    return squares, img

def clockwise(pts):
    '''将输入的点顺时针排序'''
    pts = np.array(pts)
    sort_x = pts[np.argsort(pts[:, 0]), :]
    
    Left = sort_x[:2, :]
    Right = sort_x[2:, :]
    # Left sort
    Left = Left[np.argsort(Left[:,1])[::-1], :]
    # Right sort
    Right = Right[np.argsort(Right[:,1]), :]
    res = np.concatenate((Left, Right), axis=0)

    return np.array(np.roll(res,6).tolist(), dtype=np.float32)

def target_vertax_point(clockwise_point):
    #计算顶点的宽度(取最大宽度)
    w1 = np.linalg.norm(clockwise_point[0]-clockwise_point[1])
    w2 = np.linalg.norm(clockwise_point[2]-clockwise_point[3])
    w = w1 if w1 > w2 else w2
    #计算顶点的高度(取最大高度)
    h1 = np.linalg.norm(clockwise_point[1]-clockwise_point[2])
    h2 = np.linalg.norm(clockwise_point[3]-clockwise_point[0])
    h = h1 if h1 > h2 else h2
    #将宽和高转换为整数
    w = int(round(w))
    h = int(round(h))
    #计算变换后目标的顶点坐标
    top_left = [0, 0]
    top_right = [w, 0]
    bottom_right = [w, h]
    bottom_left = [0, h]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype = np.float32)

def get_choice(circles):
    '''通过算法得出所有选项的位置'''
    x = []
    y = []
    for c in circles[0]:
        x.append(c[0])
        y.append(c[1])
    x.sort()
    y.sort()
    x_step = (x[-1] - x[0])/4
    y_step = (y[-1] - y[0])/6
    pos = (x[0], y[0])

    return pos, int(x_step), int(y_step)

class AnswerSheet:
    '''答题卡基础类'''
    def __init__(self, src, persp_img=None, canny_persp_img=None):
        self.src = src
        self.persp_img = persp_img
        self.canny_persp_img = canny_persp_img

    def PerspTrans(self):
        '''将读取的图片转换到正视角'''
        squares, self.src = find_squares(self.src)
        dots = []
        for lines in squares:
            for dot in lines:
                dots.append(dot)

        dots = clockwise(dots)
        dst_dots = target_vertax_point(dots)

        #计算变换矩阵
        matrix = cv2.getPerspectiveTransform(dots, dst_dots)
        #计算透视变换后的图片
        self.persp_img = cv2.warpPerspective(self.src, matrix, (int(dst_dots[2][0]), int(dst_dots[2][1])))
        cv2.imshow('persp_img', self.persp_img)
        
        #cv2.drawContours(self.src, squares, -1, (0, 0, 255), 2 )
    
    def ReadAns(self):
        self.canny_persp_img = cv2.Canny(self.persp_img, 30, 100)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
        self.canny_persp_img = cv2.morphologyEx(self.canny_persp_img, cv2.MORPH_CLOSE, kernel,iterations=1) 

        circles = cv2.HoughCircles(self.canny_persp_img, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=25, minRadius=0, maxRadius=50)
        circles = np.uint16(np.around(circles))  # 取整
        self.canny_persp_img = cv2.cvtColor(self.canny_persp_img, cv2.COLOR_GRAY2BGR)
        pos, x_step, y_step = get_choice(circles)
        for x in range(pos[0], pos[0] + x_step*5, x_step):
            for y in range(pos[1], pos[1] + y_step*7, y_step):
                cv2.circle(self.canny_persp_img, (x, y), 10, (0, 0, 255), 2)
            print(x)


img = cv2.imread('/Users/joecos_kun/Desktop/test.jpg')
test = AnswerSheet(img)
cv2.imshow('src', img)

test.PerspTrans()
test.ReadAns()
cv2.imshow('test', test.canny_persp_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

