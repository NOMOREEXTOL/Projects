import math as mt
import random
import numpy as np


"""
项目输入数据为，围压，含水率，动偏应力的组合数组
                (sigama3,w,sigamaD)
                每一组这种数据训练出一个W，然后与
                各自对应情况下应有的土的状态进行比较
                并对误差进行反向传播，达到一个深度学习
                的目的。
                
更深的网络比起单层隐含层具有更多节点能学到更多的特征


本例采用矩阵运算的形式进行阐释

"""


class Tools():
    """
    工具类，用于存放静态函数
    """
    @staticmethod
    def sigmod(x): # 定义激活函数
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def rand(low,high):
        return (high - low) * random.random() + low

    @staticmethod
    def meanSquareError(calcValues,labelValues):
        """
        calcValues为计算值
        labelValues为真实值
        """
        error = 0.0
        for i in range(len(calcValues)):
            for j in range(len(calcValues[0])):
                error += 0.5 * ((calcValues[i][j] - labelValues[i][j]) ** 2)
        return error



class Main:
    LEARN_RATE = 0.05 # 固定学习率
    EXPECT_ERROR = 0.0001 # 限制样本均方误差值
    MAX_CIRCULATING_TIMES = 5000 # 最大循环次数

    def __init__(self):
        """
        为了方便描述，把输入层，隐含层，输出层的神经元（节点）个数
        分别描述为IN,HN,ON
        """
        self.input_n = 0 # 输入层节点个数
        self.output_n = 0 # 隐含层节点个数
        self.hidden_n = 0 # 输出层节点个数
        self.input_values = [] # 输入层原始数据矩阵  规格为 x * IN x为实际数据量，即filePath文件行数 ，实际最后做矩阵计算的时候要进行转置
        self.output_values = [] # 输出层计算得出结果的矩阵，后与实际结果相对比。 规格为 ON * x  x为实际数据量
        self.hidden_values = [] # 隐含层数据输出矩阵，此为激活后的。 (HN * x)
        self.hidden_in_values = [] # 进入隐含层激活前的输入值矩阵，也即输入层的原始数据经过线性运算得到的矩阵(HN * x)
        self.output_in_values = [] # 进入输出层激活前的输入值矩阵，也即隐含层的输出矩阵经过线性运算得到的矩阵 (ON * x)
        self.hidden_weights = [] # 到达隐含层的权重矩阵 (HN * IN)
        self.output_weights = [] # 到达输出层的权重矩阵 (ON * HN)
        self.hidden_bias = [] # 到达隐含层的偏置矩阵 (HN * x)
        self.output_bias = [] # 到达输出层的偏置矩阵 (ON * x)
        self.output_error = [] # 输出层的偏差矩阵 (ON * x)
        self.hidden_error = [] # 隐含层的偏差矩阵(HN * x)
        self.output_label = [] # 输出层的标准输出矩阵，用于计算输出层的输出误差error,用于后续迭代，反向传播 (ON * x)



    def setup(self,filePath,HN,ON):

           # filePath为存放原始数据的文件路径，需为绝对路径  C:\Users\xiaomi\Desktop\test.txt
            #    数据存放形式为 围压 动偏应力 含水率 理论结果 每个数据以空格区分(输入数据取前三列)
            #    每行为一组数据
             #   HN 为隐含层神经元个数，现只考虑只有一层隐含层的情况
              #  ON 为输出层神经元个数

        # 初始化原始数据数组
        with open(filePath,mode='r',encoding='utf-8') as f:
            interArray1 = [i.strip().split(' ')[:-1] for i in f.readlines()]
            f.seek(0)
            interArray2 = [i.strip().split(' ')[-1:] for i in f.readlines()]
        for i in range(len(interArray1)):
            for j in range(len(interArray1[i])):
                if not interArray1[i][j]:
                    interArray1[i].remove(interArray1[i][j])
                else:
                    interArray1[i][j] = float(interArray1[i][j])
        for i in range(len(interArray2)):
            for j in range(len(interArray2[i])):
                if not interArray2[i][j]:
                    interArray2[i].remove(interArray2[i][j])
                else:
                    interArray2[i][j] = float(interArray2[i][j])
        self.output_label = np.array(interArray2).T
        self.input_values = np.array(interArray1).T

        self.input_n = len(interArray1[0])
        self.hidden_n = HN
        self.output_n = ON
        self.X = len(interArray1)

        # 初始化相关矩阵
        self.hidden_weights = np.array([[0.0 for __ in range(self.input_n)] for _ in range(self.hidden_n)])
        self.output_weights = np.array([[0.0 for __ in range(self.hidden_n)] for _ in range(self.output_n)])
        self.hidden_bias = np.array([[0.0 for __ in range(self.X)] for _ in range(self.hidden_n)])
        self.output_bias = np.array([[0.0 for __ in range(self.X)] for _ in range(self.output_n)])
        self.output_values = np.array([[0.0 for __ in range(self.X)] for _ in range(self.output_n)])
        self.hidden_values = np.array([[0.0 for __ in range(self.X)] for _ in range(self.hidden_n)])
        self.hidden_in_values = np.array([[0.0 for __ in range(self.X)] for _ in range(self.hidden_n)])
        self.output_in_values = np.array([[0.0 for __ in range(self.X)] for _ in range(self.output_n)])
        self.output_error = np.array([[0.0 for __ in range(self.X)] for _ in range(self.output_n)])
        self.hidden_error = np.array([[0.0 for __ in range(self.X)] for _ in range(self.hidden_n)])


        # 随机赋值
        random.seed(0)

        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.hidden_weights[h][i] = Tools.rand(-1,1)

        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[o][h] = Tools.rand(-1,1)


    def forward(self): # 前向传播
        # 输入层到隐含层传播
        self.hidden_in_values = self.hidden_weights @ self.input_values + self.hidden_bias
        # 隐含层激活
        for h in range(self.hidden_n):
            for j in range(self.X):
                print(self.hidden_in_values[h][j])
                self.hidden_values[h][j] = Tools.sigmod(self.hidden_in_values[h][j])
        # 隐含层到输出层
        self.output_in_values = self.output_weights @ self.hidden_values + self.output_bias
        # 输出层激活
        for o in range(self.output_n):
            for j in range(self.X):
                self.output_values[o][j] = Tools.sigmod(self.output_in_values[o][j])

        return self.output_values


    def checkError(self):
        """用于校验是否需要停止迭代"""
        currentError = Tools.meanSquareError(self.output_values,self.output_label)
        return currentError <= Main.EXPECT_ERROR


    def backPropagate(self): # 反向传播
        # 计算输出层的误差矩阵error
        self.output_error = self.output_values - self.output_label # 即sigamaL 由sigamaL可迭代求出sigamaL-1即上一层的error
        # 计算输出层的权重，偏置修正量
        self.output_weights = self.output_weights - Main.LEARN_RATE * self.output_error @ (self.hidden_values.T)
        self.output_bias = self.output_bias - Main.LEARN_RATE * self.output_error
        # 计算隐含层的等效误差
        self.hidden_error = (self.output_weights.T) @ self.output_error * self.hidden_values
        # (HN,x) = (HN,ON) @ (ON,x) * (HN,x) 矩阵的点乘与叉乘
        # 计算隐含层的权重，偏置修正量
        self.hidden_weights = self.hidden_weights - Main.LEARN_RATE * self.hidden_error @ (self.input_values.T)
        self.hidden_bias = self.hidden_bias - Main.LEARN_RATE * self.hidden_error

        # 校验样本总体均方误差是否合格，合格就退出迭代，不合格则修正w,b继续迭代


    def train(self):
        self.setup(r'C:\Users\xiaomi\Desktop\test.txt',4,1) # 输入数据
        circleTimes = 0
        self.forward() # 前向传播
        self.backPropagate() # 反向传播
        while not self.checkError() and circleTimes <= Main.MAX_CIRCULATING_TIMES:
            self.forward()  # 前向传播
            self.backPropagate()  # 反向传播
            print("当前迭代次数为{}".format(circleTimes))
            print("误差为:",end = '')
            for i in self.output_error:
                print(i,end='')
            print()
            print("当前输出值为:",end='')
            print(self.output_values)
            circleTimes += 1



if __name__ == "__main__":
    s = Main()
    s.train()

    """
    possible output can be follows:
    -1373.2949419512345
    -3.3632961514100277
    -77.4035165804202
    -91.8688115216147
    -106.3335065452646
    -91.8688115216147
    当前迭代次数为1949
    误差为:[-0.00990536 -0.00230966  0.00111321  0.00966043]
    当前输出值为:[[0.99009464 0.49769034 0.50111321 0.00966043]]
    """
