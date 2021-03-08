import numpy as np


class intersection_four:
    def __init__(self):
        self.input_w = road1
        self.input_e = road2
        self.input_s = road3
        self.input_n = road4

        self.output_w = road5
        self.output_e = road6
        self.output_s = road7
        self.output_n = road8

        self.w_rate = [0.3, 0.3, 0.3]
        self.e_rate = [0.3, 0.3, 0.3]
        self.s_rate = [0.3, 0.3, 0.3]
        self.n_rate = [0.3, 0.3, 0.3]

    def update(self):


class road:
    def __init__(self):
        self.length = 10  # 路段长度
        self.last_cell = 10  # 输入路段的最后一个cell
        self.next_cell = 10  # 输出路段的第一个cell
        self.n_ik = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # 路段元胞矩阵
        self.V_f = 25  # 自由流
        self.w = 6
        self.Q_ik = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])  # 最大流入流量
        self.N_ik = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10])  # 最大承载量
        self.y_ik = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # update中计算流入此cell的临时缓存

    def calcu_y_ik(self):
        # 逐个更新 废弃
        # for i,n_ik in enumerate(self.cells):
        #     if i==0:#如果是第一个
        #         n_ik_last=9
        #     else:
        #         n_ik_last=self.cells[i-1]
        #     self.y_ik[i]=min(n_ik_last,self.Q_ik[i],self.w/self.V_f*(self.N_ik[i]-self.n_ik[i]))
        # 逐行更新
        print("==计算yik==")
        n_ik_last = np.concatenate([[self.last_cell], self.n_ik[:-1]])
        self.y_ik = np.amin([n_ik_last, self.Q_ik, self.w / self.V_f * (self.N_ik - self.n_ik)], axis=0).astype(int)

    def update(self):
        print("==更新n_ik==")
        y_ik_next = np.concatenate([self.y_ik[1:], [0]])
        self.n_ik = self.n_ik + self.y_ik - y_ik_next


if __name__ == '__main__':
    road1 = road()
    road2 = road()
    road3 = road()
    road4 = road()
    road5 = road()
    road6 = road()
    road7 = road()
    road8 = road()
    print('更新前', road1.n_ik)
    road1.calcu_yik()
    road1.update()
    print('更新后', road1.n_ik)
