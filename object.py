from params import *

class intersection_four:
    def __init__(self, input_w, input_e, input_s, input_n, output_w, output_e, output_s, output_n, rate_func_inter):
        self.input_w = input_w
        self.input_e = input_e
        self.input_s = input_s
        self.input_n = input_n

        self.output_w = output_w
        self.output_e = output_e
        self.output_s = output_s
        self.output_n = output_n

        self.rate_func_inter=rate_func_inter


    def update(self, p_l):
        self.w_rate, self.e_rate, self.s_rate, self.n_rate = self.rate_func_inter(p_l)
        print("比例", self.w_rate, self.e_rate, self.s_rate, self.n_rate)
        self.w_rate_sum = sum(self.w_rate)
        self.e_rate_sum = sum(self.e_rate)
        self.s_rate_sum = sum(self.s_rate)
        self.n_rate_sum = sum(self.n_rate)

        if self.input_n is not None:  # 北进口
            self.input_n.output_cell = 0
            if self.output_e is not None:  # 左
                if self.n_rate[0] > 0:
                    self.input2output = np.amin(
                        [(self.output_e.l_c / self.input_n.l_i[-1]) * self.input_n.n_ik[-1] * self.n_rate[0] / self.n_rate_sum,
                         self.output_e.Q_ik[0],
                         (1 - self.output_e.gamma[0]) * self.output_e.Q_ik[0] + self.output_e.gamma[0] * (
                                 self.output_e.l_c / self.output_e.l_i[0]) * self.output_e.w / self.output_e.V_f * (
                                 self.output_e.N_ik[0] - self.output_e.n_ik[0])],
                        axis=0).astype(int)
                    self.input_n.output_cell += self.input2output
                    self.output_e.input_cell = self.input2output

            if self.output_s is not None:  # 直
                if self.n_rate[1] > 0:
                    self.input2output = np.amin(
                        [(self.output_s.l_c / self.input_n.l_i[-1]) * self.input_n.n_ik[-1] * self.n_rate[
                            1] / self.n_rate_sum,
                         self.output_s.Q_ik[0],
                         (1 - self.output_s.gamma[0]) * self.output_s.Q_ik[0] + self.output_s.gamma[0] * (
                                 self.output_s.l_c / self.output_s.l_i[0]) * self.output_s.w / self.output_s.V_f * (
                                 self.output_s.N_ik[0] - self.output_s.n_ik[0])],
                        axis=0).astype(int)
                    self.input_n.output_cell += self.input2output
                    self.output_s.input_cell = self.input2output

            if self.output_w is not None:  # 右
                if self.n_rate[2] > 0:
                    self.input2output = np.amin(
                        [(self.output_w.l_c / self.input_n.l_i[-1]) * self.input_n.n_ik[-1] * self.n_rate[
                            2] / self.n_rate_sum,
                         self.output_w.Q_ik[0],
                         (1 - self.output_w.gamma[0]) * self.output_w.Q_ik[0] + self.output_ws.gamma[0] * (
                                 self.output_w.l_c / self.output_w.l_i[0]) * self.output_w.w / self.output_w.V_f * (
                                 self.output_w.N_ik[0] - self.output_w.n_ik[0])],
                        axis=0).astype(int)
                    self.input_n.output_cell += self.input2output
                    self.output_w.input_cell = self.input2output

        if self.input_s is not None:  # 南进口
            self.input_s.output_cell = 0
            if self.output_w is not None:  # 左
                if self.s_rate[0] > 0:
                    self.input2output = np.amin(
                        [(self.output_w.l_c / self.input_s.l_i[-1]) * self.input_s.n_ik[-1] * self.s_rate[
                            0] / self.s_rate_sum,
                         self.output_w.Q_ik[0],
                         (1 - self.output_w.gamma[0]) * self.output_w.Q_ik[0] + self.output_w.gamma[0] * (
                                 self.output_w.l_c / self.output_w.l_i[0]) * self.output_w.w / self.output_w.V_f * (
                                 self.output_w.N_ik[0] - self.output_w.n_ik[0])],
                        axis=0).astype(int)
                    self.input_s.output_cell += self.input2output
                    self.output_w.input_cell = self.input2output

            if self.output_n is not None:  # 直
                if self.s_rate[1] > 0:
                    self.input2output = np.amin(
                        [(self.output_n.l_c / self.input_s.l_i[-1]) * self.input_s.n_ik[-1] * self.s_rate[
                            1] / self.s_rate_sum,
                         self.output_n.Q_ik[0],
                         (1 - self.output_n.gamma[0]) * self.output_n.Q_ik[0] + self.output_n.gamma[0] * (
                                 self.output_n.l_c / self.output_n.l_i[0]) * self.output_n.w / self.output_n.V_f * (
                                 self.output_n.N_ik[0] - self.output_n.n_ik[0])],
                        axis=0).astype(int)
                    self.input_s.output_cell += self.input2output
                    self.output_n.input_cell = self.input2output

            if self.output_e is not None:  # 右
                if self.s_rate[2] > 0:
                    self.input2output = np.amin(
                        [(self.output_e.l_c / self.input_s.l_i[-1]) * self.input_s.n_ik[-1] * self.s_rate[
                            2] / self.s_rate_sum,
                         self.output_e.Q_ik[0],
                         (1 - self.output_e.gamma[0]) * self.output_e.Q_ik[0] + self.output_e.gamma[0] * (
                                 self.output_e.l_c / self.output_e.l_i[0]) * self.output_e.w / self.output_e.V_f * (
                                 self.output_e.N_ik[0] - self.output_e.n_ik[0])],
                        axis=0).astype(int)
                    self.input_s.output_cell += self.input2output
                    self.output_e.input_cell = self.input2output

        if self.input_w is not None:  # 西进口
            self.input_w.output_cell = 0
            if self.output_n is not None:  # 左
                if self.w_rate[0] > 0:
                    self.input2output = np.amin(
                        [(self.output_n.l_c / self.input_w.l_i[-1]) * self.input_w.n_ik[-1] * self.w_rate[
                            0] / self.w_rate_sum,
                         self.output_n.Q_ik[0],
                         (1 - self.output_n.gamma[0]) * self.output_n.Q_ik[0] + self.output_n.gamma[0] * (
                                 self.output_n.l_c / self.output_n.l_i[0]) * self.output_n.w / self.output_n.V_f * (
                                 self.output_n.N_ik[0] - self.output_n.n_ik[0])],
                        axis=0).astype(int)
                    self.input_w.output_cell += self.input2output
                    self.output_n.input_cell = self.input2output

            if self.output_e is not None:  # 直
                if self.w_rate[1] > 0:
                    self.input2output = np.amin(
                        [(self.output_e.l_c / self.input_w.l_i[-1]) * self.input_w.n_ik[-1] * self.w_rate[
                            1] / self.w_rate_sum,
                         self.output_e.Q_ik[0],
                         (1 - self.output_e.gamma[0]) * self.output_e.Q_ik[0] + self.output_e.gamma[0] * (
                                 self.output_e.l_c / self.output_e.l_i[0]) * self.output_e.w / self.output_e.V_f * (
                                 self.output_e.N_ik[0] - self.output_e.n_ik[0])],
                        axis=0).astype(int)
                    self.input_w.output_cell += self.input2output
                    self.output_e.input_cell = self.input2output

            if self.output_s is not None:  # 右
                if self.w_rate[2] > 0:
                    self.input2output = np.amin(
                        [(self.output_s.l_c / self.input_w.l_i[-1]) * self.input_w.n_ik[-1] * self.w_rate[
                            2] / self.w_rate_sum,
                         self.output_s.Q_ik[0],
                         (1 - self.output_s.gamma[0]) * self.output_s.Q_ik[0] + self.output_s.gamma[0] * (
                                 self.output_s.l_c / self.output_s.l_i[0]) * self.output_s.w / self.output_s.V_f * (
                                 self.output_s.N_ik[0] - self.output_s.n_ik[0])],
                        axis=0).astype(int)
                    self.input_w.output_cell += self.input2output
                    self.output_s.input_cell = self.input2output

        if self.input_e is not None:  # 东进口
            self.input_e.output_cell = 0
            if self.output_s is not None:  # 左
                if self.e_rate[0] > 0:
                    self.input2output = np.amin(
                        [(self.output_s.l_c / self.input_e.l_i[-1]) * self.input_e.n_ik[-1] * self.e_rate[
                            0] / self.e_rate_sum,
                         self.output_s.Q_ik[0],
                         (1 - self.output_s.gamma[0]) * self.output_s.Q_ik[0] + self.output_s.gamma[0] * (
                                 self.output_s.l_c / self.output_s.l_i[0]) * self.output_s.w / self.output_s.V_f * (
                                 self.output_s.N_ik[0] - self.output_s.n_ik[0])],
                        axis=0).astype(int)
                    self.input_e.output_cell += self.input2output
                    self.output_s.input_cell = self.input2output

            if self.output_w is not None:  # 直
                if self.e_rate[1] > 0:
                    self.input2output = np.amin(
                        [(self.output_w.l_c / self.input_w.l_i[-1]) * self.input_w.n_ik[-1] * self.e_rate[
                            1] / self.e_rate_sum,
                         self.output_w.Q_ik[0],
                         (1 - self.output_w.gamma[0]) * self.output_w.Q_ik[0] + self.output_w.gamma[0] * (
                                 self.output_w.l_c / self.output_w.l_i[0]) * self.output_w.w / self.output_w.V_f * (
                                 self.output_w.N_ik[0] - self.output_w.n_ik[0])],
                        axis=0).astype(int)
                    self.input_e.output_cell += self.input2output
                    self.output_w.input_cell = self.input2output

            if self.output_n is not None:  # 右
                if self.e_rate[2] > 0:
                    self.input2output = np.amin(
                        [(self.output_n.l_c / self.input_e.l_i[-1]) * self.input_e.n_ik[-1] * self.e_rate[
                            2] / self.e_rate_sum,
                         self.output_n.Q_ik[0],
                         (1 - self.output_n.gamma[0]) * self.output_n.Q_ik[0] + self.output_n.gamma[0] * (
                                 self.output_n.l_c / self.output_n.l_i[0]) * self.output_n.w / self.output_n.V_f * (
                                 self.output_n.N_ik[0] - self.output_n.n_ik[0])],
                        axis=0).astype(int)
                    self.input_e.output_cell += self.input2output
                    self.output_n.input_cell = self.input2output


class road:
    def __init__(self, road_set):
        self.length = road_set["length"]  # 路段长度
        self.input_cell = None  # 输入路段的最后一个cell
        self.output_cell = None  # 输出路段的第一个cell
        self.n_ik = np.array([0] * self.length)  # 路段元胞矩阵
        self.V_f = 14  # 自由流
        self.w = 6

        self.Q_ik = road_set["Q_ik"]  # 最大流入流量
        self.N_ik = road_set["N_ik"]  # 最大承载量
        self.y_ik = np.array([0] * self.length)  # update中计算流入此cell的临时缓存
        self.l_c = 70
        self.l_i = road_set["l_i"]

        self.every_epoch2time = 5
        self.d_T = self.every_epoch2time
        self.rou_j = 0.399
        self.rou_c1 = 0

        self.gamma = np.array([0] * self.length)

        # self.gamma  # 判断函数

    def calcu_gamma_ik(self):
        gamma_new = []
        for index, n in enumerate(self.n_ik):
            if n <= self.N_i_c1[index]:
                gamma_new.append(0)
            elif n >= self.N_i_c2[index]:
                gamma_new.append(1)
            else:
                gamma_new.append(self.gamma[index])
        self.gamma = gamma_new

    def calcu_N_c1_c2(self):
        self.N_i_c1 = np.array([self.d_T * self.rou_c1] * self.length).astype(int)
        # print("N_i_c1", self.N_i_c1)
        self.N_i_c2 = (self.Q_ik / self.V_f).astype(int)
        # print("N_i_c2", self.N_i_c2)

    def calcu_y_ik(self):
        # 逐个更新 废弃
        # for i,n_ik in enumerate(self.cells):
        #     if i==0:#如果是第一个
        #         n_ik_last=9
        #     else:
        #         n_ik_last=self.cells[i-1]
        #     self.y_ik[i]=min(n_ik_last,self.Q_ik[i],self.w/self.V_f*(self.N_ik[i]-self.n_ik[i]))
        # 逐行更新

        # print("==计算yik==")
        n_ik_last = np.concatenate([[0], self.n_ik[:-1]])  # 假设上一段输入为0，后面再补上
        # print(self.l_i)
        self.l_i_previous = np.concatenate([[80], self.l_i[:-1]])  # 这里留了一个问题，这里80应该是变量，但是由于先假设上一路段输出为零，所以这里80不会参与有效的计算
        # print(self.l_i_previous)
        self.y_ik = np.amin([(self.l_c / self.l_i_previous) * n_ik_last, self.Q_ik,
                             (np.array([1] * self.length) - self.gamma) * self.Q_ik + self.gamma * (
                                     self.l_c / self.l_i) * self.w / self.V_f * (self.N_ik - self.n_ik)],
                            axis=0).astype(int)

    def update(self):
        self.calcu_N_c1_c2()
        self.calcu_gamma_ik()
        self.calcu_y_ik()
        print("input",self.input_cell)
        print("output", self.output_cell)
        print("==更新n_ik==")
        y_ik_next = np.concatenate([self.y_ik[1:], [0]])
        self.n_ik = self.n_ik + self.y_ik - y_ik_next
        if self.input_cell is not None:
            self.n_ik[0] += self.input_cell
        if self.output_cell is not None:
            self.n_ik[-1] -= self.output_cell
        print("n_ik",self.n_ik)


road_sets = [{"length": 3, "l_i_last": 100},  # input_road
             {"length": 21, "l_i_last": 100},
             {"length": 17, "l_i_last": 85},
             {"length": 31, "l_i_last": 80},
             {"length": 39, "l_i_last": 110},
             {"length": 33, "l_i_last": 75},
             {"length": 21, "l_i_last": 100},
             {"length": 25, "l_i_last": 100},
             {"length": 30, "l_i_last": 90},
             {"length": 34, "l_i_last": 110},
             {"length": 31, "l_i_last": 80},
             {"length": 15, "l_i_last": 90},
             {"length": 17, "l_i_last": 80},
             {"length": 3, "l_i_last": 80}]  # output_road

for road_set in road_sets:
    road_set["Q_ik"] = np.array([7] * road_set["length"])
    road_set["l_i"] = np.array([70] * (road_set["length"] - 1) + [road_set["l_i_last"]])
    # road_set["N_ik"] = road_set["Q_ik"] / 70 * road_set["l_i"].astype(int)
    road_set["N_ik"] = 0.399 * road_set["l_i"].astype(int)

road_in = road(road_sets[0])
road_in.N_ik[0]=10000#让输入路段缓存量很大，防止堵塞再路段外
road1 = road(road_sets[1])
road2 = road(road_sets[2])
road3 = road(road_sets[3])
road4 = road(road_sets[4])
road5 = road(road_sets[5])
road6 = road(road_sets[6])
road7 = road(road_sets[7])
road8 = road(road_sets[8])
road9 = road(road_sets[9])
road10 = road(road_sets[10])
road11 = road(road_sets[11])
road12 = road(road_sets[12])
road_out = road(road_sets[13])

def rate_func_inter_1(p_l):
    w_rate = [0, 0, 0]
    e_rate = [0, 0, 0]
    s_rate = [0, 0, 0]
    n_rate = [p_l[0] + p_l[1] + p_l[2], p_l[3] + p_l[4] + p_l[5], 0]
    return w_rate, e_rate, s_rate, n_rate


def rate_func_inter_2(p_l):
    w_rate = [0, p_l[0], p_l[1] + p_l[2]]
    e_rate = [0, 0, 0]
    s_rate = [0, 0, 0]
    n_rate = [0, 0, 0]
    return w_rate, e_rate, s_rate, n_rate


def rate_func_inter_3(p_l):
    w_rate = [0, 0, 1]
    e_rate = [0, 0, 0]
    s_rate = [0, 0, 0]
    n_rate = [0, 0, 0]
    return w_rate, e_rate, s_rate, n_rate


def rate_func_inter_4(p_l):
    w_rate = [0, 0, 0]
    e_rate = [0, 0, 0]
    s_rate = [0, 0, 0]
    n_rate = [p_l[3] + p_l[4], p_l[5], 0]
    return w_rate, e_rate, s_rate, n_rate


def rate_func_inter_5(p_l):
    w_rate = [0, p_l[3], p_l[4]]
    e_rate = [0, 0, 0]
    s_rate = [0, 0, 0]
    n_rate = [p_l[1], p_l[2], 0]
    return w_rate, e_rate, s_rate, n_rate


def rate_func_inter_6(p_l):
    w_rate = [0, 0, 1]
    e_rate = [0, 0, 0]
    s_rate = [0, 0, 0]
    n_rate = [0, 1, 0]
    return w_rate, e_rate, s_rate, n_rate


def rate_func_inter_7(p_l):
    w_rate = [0, 0, 0]
    e_rate = [0, 0, 0]
    s_rate = [0, 0, 0]
    n_rate = [1, 0, 0]
    return w_rate, e_rate, s_rate, n_rate


def rate_func_inter_8(p_l):
    w_rate = [0, 1, 0]
    e_rate = [0, 0, 0]
    s_rate = [0, 0, 0]
    n_rate = [1, 0, 0]
    return w_rate, e_rate, s_rate, n_rate


def rate_func_inter_9(p_l):
    w_rate = [0, 0, 1]
    e_rate = [0, 0, 0]
    s_rate = [0, 0, 0]
    n_rate = [0, 1, 0]
    return w_rate, e_rate, s_rate, n_rate


input_w = None
input_e = None
input_s = None
input_n = road_in

output_w = None
output_e = road1
output_s = road3
output_n = None

inter1 = intersection_four(input_w, input_e, input_s, input_n, output_w, output_e, output_s, output_n,
                           rate_func_inter_1)

input_w = road1
input_e = None
input_s = None
input_n = None

output_w = None
output_e = road2
output_s = road4
output_n = None

inter2 = intersection_four(input_w, input_e, input_s, input_n, output_w, output_e, output_s, output_n,
                           rate_func_inter_2)

input_w = road2
input_e = None
input_s = None
input_n = None

output_w = None
output_e = None
output_s = road5
output_n = None

inter3 = intersection_four(input_w, input_e, input_s, input_n, output_w, output_e, output_s, output_n,
                           rate_func_inter_3)

input_w = None
input_e = None
input_s = None
input_n = road3

output_w = None
output_e = road6
output_s = road8
output_n = None

inter4 = intersection_four(input_w, input_e, input_s, input_n, output_w, output_e, output_s, output_n,
                           rate_func_inter_4)

input_w = road6
input_e = None
input_s = None
input_n = road4

output_w = None
output_e = road7
output_s = road9
output_n = None

inter5 = intersection_four(input_w, input_e, input_s, input_n, output_w, output_e, output_s, output_n,
                           rate_func_inter_5)

input_w = road7
input_e = None
input_s = None
input_n = road5

output_w = None
output_e = None
output_s = road10
output_n = None

inter6 = intersection_four(input_w, input_e, input_s, input_n, output_w, output_e, output_s, output_n,
                           rate_func_inter_6)

input_w = None
input_e = None
input_s = None
input_n = road8

output_w = None
output_e = road11
output_s = None
output_n = None

inter7 = intersection_four(input_w, input_e, input_s, input_n, output_w, output_e, output_s, output_n,
                           rate_func_inter_7)

input_w = road11
input_e = None
input_s = None
input_n = road9

output_w = None
output_e = road12
output_s = None
output_n = None

inter8 = intersection_four(input_w, input_e, input_s, input_n, output_w, output_e, output_s, output_n,
                           rate_func_inter_8)

input_w = road12
input_e = None
input_s = None
input_n = road10

output_w = None
output_e = None
output_s = road_out
output_n = None

inter9 = intersection_four(input_w, input_e, input_s, input_n, output_w, output_e, output_s, output_n,
                           rate_func_inter_9)



