import cv2
import numpy as np
from matplotlib import pyplot as plt

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
    x_step = int((x[-1] - x[0])/4)
    y_step = int((y[-1] - y[0])/6)
    pos = (x[0], y[0])

    #将计算出的结果存放在choices中
    choices = []
    for y in range(pos[1], pos[1] + y_step*7, y_step):
        for x in range(pos[0], pos[0] + x_step*5, x_step):
            choices.append((x, y))

    return choices, x_step, y_step

def find_target(target, template, size):
    output = []
    template = cv2.resize(template, size)
    #cv2.imshow('template', template)
    theight, twidth = template.shape[:2]

    #执行模板匹配，采用的匹配方式cv2.TM_SQDIFF_NORMED
    result = cv2.matchTemplate(target, template, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    #绘制矩形边框，将匹配区域标注出来
    output.append(min_loc)

    #初始化位置参数
    temp_loc = min_loc
    other_loc = min_loc
    numOfloc = 1

    #第一次筛选----规定匹配阈值，将满足阈值的从result中提取出来
    threshold = 0.4
    loc = np.where(result < threshold)

    #遍历提取出来的位置, 初次将位置偏移小于20个像素的结果舍去
    for other_loc in zip(*loc[::-1]):
        if (temp_loc[0]+20<other_loc[0])or(temp_loc[1]+20<other_loc[1]):
            numOfloc = numOfloc + 1
            temp_loc = other_loc
            output.append(other_loc)

    #再次舍去位置偏移小于10的结果，并按y轴排序
    output = kick(output)
    output = sorted(output, key = lambda x:(x[1]))
    for pos in output:
        cv2.rectangle(target,pos, (pos[0]+twidth,pos[1]+theight), (0,0,225), 2)
    return output, target

def draw_circles(choices, img):
    for pos in choices:
        cv2.circle(img, pos, 10, (0, 0, 255), 2)
    return img

def kick(targets):
    output = []
    for i in range(len(targets)-1):
        distance = []
        for j in range(i+1, len(targets)):
            distance.append(abs(targets[i][0] - targets[j][0]) + abs(targets[i][1] - targets[j][1]))
        distance.sort()
        if distance.pop(0)> 10:
            output.append(targets[i])
    output.append(targets[-1])
    return output

class AnswerSheet:
    '''答题卡基础类'''
    def __init__(self, src, sheet_name, persp_img=None, canny_persp_img=None, correct_ans=['A', 'B', 'C', 'D', 'E', 'A', 'B']):
        self.src = src
        self.sheet_name = sheet_name
        self.persp_img = persp_img
        self.canny_persp_img = canny_persp_img
        self.target_img = cv2.imread('target.jpg')
        self.correct_ans = correct_ans

    def PerspTrans(self, ShowProgress=0):
        '''将读取的图片转换到正视角'''

        #找到图片中的凸四边形轮廓，获取其四个顶点并按顺时针正确排列
        squares, self.src = find_squares(self.src)
        dots = []
        for lines in squares:
            for dot in lines:
                dots.append(dot)
        dots = clockwise(dots)
        dst_dots = target_vertax_point(dots)

        #计算变换矩阵
        matrix = cv2.getPerspectiveTransform(dots, dst_dots)

        #计算透视变换后的图片，并修改为合适的大小
        self.persp_img = cv2.warpPerspective(self.src, matrix, (int(dst_dots[2][0]), int(dst_dots[2][1])))
        self.persp_img = cv2.resize(self.persp_img, (int(self.persp_img.shape[1] * (435/self.persp_img.shape[0])), 435))

        if ShowProgress:
            cv2.imshow('persp_img', self.persp_img)
        
        #cv2.drawContours(self.src, squares, -1, (0, 0, 255), 2 )
    
    def ReadAns(self, ShowProgress=0):
        '从透视变换后的图像中读取答案并批改'
        self.canny_persp_img = cv2.Canny(self.persp_img, 30, 100)

        #做闭运算，使图中的圆更容易被检测
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
        self.canny_persp_img = cv2.morphologyEx(self.canny_persp_img, cv2.MORPH_CLOSE, kernel,iterations=1) 

        #进行霍夫圆检测
        circles = cv2.HoughCircles(self.canny_persp_img, cv2.HOUGH_GRADIENT, 1, 20, param1=150, param2=30, minRadius=0, maxRadius=50)
        circles = np.uint16(np.around(circles))  # 取整
        self.canny_persp_img = cv2.cvtColor(self.canny_persp_img, cv2.COLOR_GRAY2BGR)

        #通过获取的圆的数据，计算出每一个选项的大致位置
        positions, x_step, y_step = get_choice(circles)
        targets, self.persp_img = find_target(self.persp_img, self.target_img, (int(x_step/1.2), int(y_step/1.2)))

        if ShowProgress:
            self.canny_persp_img = draw_circles(positions, self.canny_persp_img)
            cv2.imshow('progress2', self.canny_persp_img)
            cv2.imshow('progress3', self.persp_img)

        #如果找到的目标不是7个，则终止运行
        if len(targets) != 7:
            print('found ' + str(len(targets)) + ' targets')
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            exit()

        #通过匹配到的涂黑的位置，计算出选择的选项
        answers = []
        alphabet = ['A', 'B', 'C', 'D', 'E']
        for i in range(7):
            for j in range(5):
                if targets[i][0] < positions[i*5 + j][0]:
                    answers.append(alphabet[j])
                    break
        #print(answers)

        #将不正确的题号储存，并在题号附近标注这题的对错
        incorrect = []
        for i in range(7):
            if answers[i] != self.correct_ans[i]:
                incorrect.append(i)
                cv2.putText(self.persp_img, 'False', (positions[0][0]-x_step*3, positions[5*i][1]), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)
            else:
                cv2.putText(self.persp_img, 'True', (positions[0][0]-x_step*3, positions[5*i][1]), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)
        
        #将批改数据打印在self.persp_img中
        cv2.putText(self.persp_img, 'correct answer:' + str(self.correct_ans), (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(self.persp_img, 'your answer:' + str(answers), (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(self.persp_img, 'Accuracy:' + str(7-len(incorrect)) + '/7', (10, 40), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)
        #cv2.imshow(self.sheet_name, self.persp_img)

def main():
    for i in range(3):
        #读取图片
        img = cv2.imread('./sheets/sheet' + str(i+1) + '.jpg')
        sheet = AnswerSheet(img, 'sheet-' + str(i))

        #对答题卡进行读取和批改
        sheet.PerspTrans(ShowProgress=0)
        sheet.ReadAns(ShowProgress=0)

        #转换图片格式并用matplotlib展示
        pic1= cv2.cvtColor(sheet.src, cv2.COLOR_BGR2RGB)
        pic2= cv2.cvtColor(sheet.persp_img, cv2.COLOR_BGR2RGB)
        fig = plt.figure(figsize=(10, 5))
        fig.canvas.manager.set_window_title('Close this window to continue.')

        # 在第一个子图中展示第一张图片
        plt.subplot(1, 2, 1)
        plt.imshow(pic1)
        plt.title('Original sheet')
        plt.axis('off')

        # 在第二个子图中展示第二张图片
        plt.subplot(1, 2, 2)
        plt.imshow(pic2)
        plt.title('Marked sheet')
        plt.axis('off')
        
        plt.show()
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

if __name__ == '__main__':
    main()