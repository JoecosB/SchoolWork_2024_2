err = 0
try:
    import cv2
except ModuleNotFoundError:
    err = 1
    print("Module opencv-python needed, but not found.")

try:
    import numpy as np
except ModuleNotFoundError:
    err = 1
    print("Module numpy needed, but not found.")

try:
    from matplotlib import pyplot as plt
except ModuleNotFoundError:
    err = 1
    print("Module matplotlib needed, but not found.")

try:
    open("./sheets/sheet1.jpg")
except FileNotFoundError:
    err = 1
    print("No sheet detected under ./sheets directory, or wrongly named.")

if err:
    exit()


class AnswerSheet:
    '''答题卡基础类'''

    def __init__(self, src, sheet_name, correct_ans=['A', 'B', 'C', 'D', 'E', 'A', 'B']):
        self.src = src
        self.sheet_name = sheet_name
        self.correct_ans = correct_ans
        self.contour = self.get_contour()
        self.persp_img = PerspImg()
        self.make_persp_img()


    @staticmethod
    def angle_cos(p0, p1, p2):
        '''计算cos值'''
        d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
        return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )
    
    
    @staticmethod
    def find_squares(img):
        '''找到图片中的矩形轮廓'''
        squares = []
        img = cv2.GaussianBlur(img, (3, 3), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bin = cv2.Canny(gray, 30, 100, apertureSize=3)    
        contours, _hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        index = 0

        # 轮廓遍历
        for cnt in contours:

            # 计算轮廓周长，并用多边形逼近
            cnt_len = cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)

            # 条件判断逼近边的数量是否为4，轮廓面积是否大于1000，检测轮廓是否为凸的
            if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):                
                cnt = cnt.reshape(-1, 2)
                squares.append(cnt)
        return squares, img


    @staticmethod
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


    @staticmethod
    def target_vertax_point(clockwise_point):
        '''计算透视变换后的点的坐标'''
        
        # 计算顶点的宽度(取最大宽度)
        w1 = np.linalg.norm(clockwise_point[0]-clockwise_point[1])
        w2 = np.linalg.norm(clockwise_point[2]-clockwise_point[3])
        w = w1 if w1 > w2 else w2

        # 计算顶点的高度(取最大高度)
        h1 = np.linalg.norm(clockwise_point[1]-clockwise_point[2])
        h2 = np.linalg.norm(clockwise_point[3]-clockwise_point[0])
        h = h1 if h1 > h2 else h2

        # 将宽和高转换为整数
        w = int(round(w))
        h = int(round(h))

        # 计算变换后目标的顶点坐标
        top_left = [0, 0]
        top_right = [w, 0]
        bottom_right = [w, h]
        bottom_left = [0, h]
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype = np.float32)


    @staticmethod
    def draw_circles(choices, img):
        '''一次性画很多圆'''
        for pos in choices:
            cv2.circle(img, pos, 10, (0, 0, 255), 2)
        return img


    def get_contour(self):
        '''找到图片中的矩形轮廓'''

        # 图像预处理
        img = cv2.GaussianBlur(self.src, (3, 3), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bin = cv2.Canny(gray, 30, 100, apertureSize=3)    
        contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 定义列表用来封装结果
        squares = []

        # 轮廓遍历
        for cnt in contours:

            # 计算轮廓周长，并且进行多边形逼近
            cnt_len = cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)

            # 条件判断逼近边的数量是否为4，轮廓面积是否大于1000，检测轮廓是否为凸的
            if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):

                cnt = cnt.reshape(-1, 2)
                squares.append(cnt)
        
        return squares


    def make_persp_img(self):
        '''创建透视变换后的图像'''

        # 获取轮廓四个顶点并按顺时针正确排列
        dots = []
        for dot in self.contour[0]:
            dots.append(dot)
        dots = AnswerSheet.clockwise(dots)

        #计算出透视变换之后的顶点
        dst_dots = AnswerSheet.target_vertax_point(dots)

        # 计算变换矩阵
        matrix = cv2.getPerspectiveTransform(dots, dst_dots)

        # 计算透视变换后的图片，并修改为合适的大小
        img = cv2.GaussianBlur(self.src, (3, 3), 0)
        img = cv2.warpPerspective(img, matrix, (int(dst_dots[2][0]), int(dst_dots[2][1])))
        img = cv2.resize(img, (int(img.shape[1] * (435/img.shape[0])), 435))

        # 生成PerspImg类属性
        self.persp_img.src = img
        self.persp_img.correct = self.correct_ans
        self.persp_img.choices = self.persp_img.get_choices()
        self.persp_img.filled = self.persp_img.get_filled()
        self.persp_img.marked = self.persp_img.mark()

class PerspImg:
    '''透视变换后的图像类'''

    def __init__(self):
        self.src = None
        self.choices = None
        self.filled = None
        self.marked = None
        self.correct = None
        self.target_img = cv2.imread('target.jpg')


    @staticmethod
    def get_choice(circles):
        '''通过算法得出所有选项的位置'''

        # 计算得出选项之间的跨度
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

        # 将计算出的结果存放在choices中
        choices = []
        for y in range(pos[1], pos[1] + y_step*7, y_step):
            for x in range(pos[0], pos[0] + x_step*5, x_step):
                choices.append((x, y))

        return choices, x_step, y_step


    @staticmethod
    def kick(targets):
        '''筛除少量重复的结果'''

        output = []

        # 遍历匹配结果
        for i in range(len(targets)-1):

            #定义一个列表，储存一个匹配结果和其他匹配结果之间的距离
            distance = []
            for j in range(i+1, len(targets)):
                distance.append(abs(targets[i][0] - targets[j][0]) + abs(targets[i][1] - targets[j][1]))

            # 将列表排序，获取距离的最小值，若最小值小于10，则剔除这个结果
            distance.sort()
            if distance.pop(0)> 10:
                output.append(targets[i])
        output.append(targets[-1])

        return output
    

    @staticmethod
    def find_target(target, template, size):
        '''对填涂选项进行模板匹配和结果筛选'''
        output = []
        template = cv2.resize(template, size)
        #cv2.imshow('template', template)
        theight, twidth = template.shape[:2]

        # 执行模板匹配，采用的匹配方式cv2.TM_SQDIFF_NORMED
        result = cv2.matchTemplate(target, template, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # 绘制矩形边框，将匹配区域标注出来
        output.append(min_loc)

        # 初始化位置参数
        temp_loc = min_loc
        other_loc = min_loc
        numOfloc = 1

        # 第一次筛选----规定匹配阈值，将满足阈值的从result中提取出来
        threshold = 0.4
        loc = np.where(result < threshold)

        # 遍历提取出来的位置, 初次将位置偏移小于20个像素的结果舍去
        for other_loc in zip(*loc[::-1]):
            if (temp_loc[0]+20<other_loc[0])or(temp_loc[1]+20<other_loc[1]):
                numOfloc = numOfloc + 1
                temp_loc = other_loc
                output.append(other_loc)

        # 再次舍去位置偏移小于10的结果，并按y轴排序
        output = PerspImg.kick(output)
        output = sorted(output, key = lambda x:(x[1]))
        for pos in output:
            cv2.rectangle(target,pos, (pos[0]+twidth,pos[1]+theight), (0,0,225), 2)
        return output
    

    def get_choices(self):
        '''获取答题卡中所有选项的位置'''
        cannyed = cv2.Canny(self.src, 30, 100)

        # 做闭运算，使图中的圆更容易被检测
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
        cannyed = cv2.morphologyEx(cannyed, cv2.MORPH_CLOSE, kernel,iterations=1) 

        # 进行霍夫圆检测
        circles = cv2.HoughCircles(cannyed, cv2.HOUGH_GRADIENT, 1, 20, param1=150, param2=30, minRadius=0, maxRadius=50)
        circles = np.uint16(np.around(circles))  # 取整

        # 通过获取的圆的数据，计算出每一个选项的大致位置
        positions, x_step, y_step = PerspImg.get_choice(circles)

        return [positions, x_step, y_step]


    def get_filled(self):
        '''获取答题卡中所有已填涂的选项'''

        # 匹配到填涂答案
        positions , x_step, y_step = self.choices
        targets = PerspImg.find_target(self.src, self.target_img, (int(x_step/1.2), int(y_step/1.2)))

        # 如果找到的目标不是7个，则终止运行
        if len(targets) != 7:
            print(f'found {len(targets)} targets.')
            AnswerSheet.draw_circles(positions, self.src)
            cv2.imshow('1', self.src)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            exit()
        
        # 通过匹配到的涂黑的位置，判断出选择的选项
        answers = []
        alphabet = ['A', 'B', 'C', 'D', 'E']
        for i in range(7):
            for j in range(5):

                # 判断选择的是哪个选项
                if targets[i][0] < positions[i*5 + j][0]:
                    answers.append(alphabet[j])
                    break

        return answers


    def mark(self):
        '''批改'''

        positions , x_step, _ = self.choices
        img = self.src.copy()

        # 将不正确的题号储存，并在题号附近标注这题的对错
        incorrect = []
        for i in range(7):
            if self.filled[i] != self.correct[i]:
                incorrect.append(i)
                cv2.putText(img, 'False', (positions[0][0]-x_step*3, positions[5*i][1]), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)
            else:
                cv2.putText(img, 'True', (positions[0][0]-x_step*3, positions[5*i][1]), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)
        
        # 将批改数据打印在self.persp_img中
        cv2.putText(img, 'correct answer:' + str(self.correct), (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(img, 'your answer:' + str(self.filled), (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(img, 'Accuracy:' + str(7-len(incorrect)) + '/7', (10, 40), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)

        return img


if __name__ == '__main__':

    def main():
        for i in range(3):
            # 读取图片
            img = cv2.imread(f'./sheets/sheet{str(i+1)}.jpg')
            sheet = AnswerSheet(img, f'sheet-{str(i)}')

            # 转换图片格式并用matplotlib展示
            pic1= cv2.cvtColor(sheet.src, cv2.COLOR_BGR2RGB)
            pic2= cv2.cvtColor(sheet.persp_img.marked, cv2.COLOR_BGR2RGB)
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

    main()