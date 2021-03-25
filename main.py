import cv2
from matplotlib import pyplot as plt

from object import *
from params import *

step = 0
input_cars_num = np.random.poisson(lam=8, size=epoch_num)

total_inputcar_num = 0
total_outputcar_num = 0
p_l = p_l0

jam_cars_num_all=[]

road4_Q_ik_back = []
road4_N_ik_back = []
for step in range(epoch_num):

    print("放入车辆", input_cars_num[step])
    # 给输入段车辆
    total_inputcar_num += input_cars_num[step]
    road_in.n_ik[0] += input_cars_num[step]
    print("+++++++++inter_1")
    inter1.update(p_l)
    print("+++++++++inter_2")
    inter2.update(p_l)
    print("+++++++++inter_3")
    inter3.update(p_l)
    print("+++++++++inter_4")
    inter4.update(p_l)
    print("+++++++++inter_5")
    inter5.update(p_l)
    print("+++++++++inter_6")
    inter6.update(p_l)
    print("+++++++++inter_7")
    inter7.update(p_l)
    print("+++++++++inter_8")
    inter8.update(p_l)
    print("+++++++++inter_9")
    inter9.update(p_l)
    print("+++++++++road_in")
    road_in.update()
    print("+++++++++road_1")
    road1.update()
    print("+++++++++road_2")
    road2.update()
    print("+++++++++road_3")
    road3.update()
    print("+++++++++road_4")
    road4.update()
    print("+++++++++road_5")
    road5.update()
    print("+++++++++road_6")
    road6.update()
    print("+++++++++road_7")
    road7.update()
    print("+++++++++road_8")
    road8.update()
    print("+++++++++road_9")
    road9.update()
    print("+++++++++road_10")
    road10.update()
    print("+++++++++road_11")
    road11.update()
    print("+++++++++road_12")
    road12.update()
    print("+++++++++road_out")
    road_out.update()
    print("==============step==============", step)
    img_cells = np.zeros([79, 46], np.uint8)
    img_cells[0:3, 0] = road_in.n_ik.T / 7 * 255
    img_cells[3:31 + 3, 0] = road3.n_ik.T / 7 * 255
    img_cells[3 + 39:3 + 39 + 30, 0] = road8.n_ik.T / 7 * 255

    img_cells[3:3 + 39, 21] = road4.n_ik.T / 7 * 255
    img_cells[3 + 39:3 + 39 + 34, 21] = road9.n_ik.T / 7 * 255

    img_cells[3:3 + 33, -1] = road5.n_ik.T / 7 * 255
    img_cells[3 + 39:3 + 39 + 31, -1] = road10.n_ik.T / 7 * 255
    img_cells[3 + 39 + 34:3 + 39 + 34 + 3, -1] = road_out.n_ik.T / 7 * 255

    img_cells[3, 0:21] = road1.n_ik / 7 * 255
    img_cells[3, 21:21 + 17] = road2.n_ik / 7 * 255

    img_cells[3 + 39, 0:21] = road6.n_ik / 7 * 255
    img_cells[3 + 39, 21:21 + 25] = road7.n_ik.T / 7 * 255

    img_cells[3 + 39 + 34, 0:15] = road11.n_ik / 7 * 255
    img_cells[3 + 39 + 34, 21:21 + 17] = road12.n_ik / 7 * 255

    img_cells = cv2.resize(img_cells, (0, 0), fx=9, fy=9, interpolation=cv2.INTER_NEAREST)

    total_outputcar_num += road_out.n_ik[0]
    road_out.n_ik[0] = 0

    cv2.imshow("img", img_cells)
    cv2.waitKey(1)


    road_list = [road1, road2, road3, road4, road5, road6, road7, road8, road9, road10, road11, road12]
    x_a = np.array([road.y_ik[1] for road in road_list])
    t_a = np.array(
        [t_a0[index] * (1 + (x_a_i / Q) ** (1.88 + 7 * (x_a_i / Q) ** 3)) for index, x_a_i in enumerate(x_a)])
    t_l = np.dot(t_a, delta)
    t_l_eqaul = np.mean(t_l)
    t_exp = np.exp(-3.3 * t_l / t_l_eqaul)  # 中间量
    p_l_new = np.array(t_exp / np.sum(t_exp))
    print("p_l", p_l_new)
    p_l = p_l_new

    jam_cars_num = 0
    for road in road_list:
        panduan_list = road.y_ik/ 5
        for panduan in panduan_list:
            if panduan > 0.9 * 0.399:
                jam_cars_num += 1

    late_cars_num = 0
    for road in road_list:
        print(road.n_ik)
        y_ik_next = np.concatenate([road.y_ik[1:], [road.n_ik[-1]]])
        print(y_ik_next)
        late_cars_num += np.sum(road.n_ik - y_ik_next)
    print("total input{}  \toutput{}".format(total_inputcar_num, total_outputcar_num))
    print("jam_cars_num{}".format(jam_cars_num))
    jam_cars_num_all.append(jam_cars_num)
    print("late_cars_num{}".format(late_cars_num))

    if step == 30 * 60 / 5:  # 如果在30分钟
        road4_N_ik_back = [road4.N_ik[33], road4.N_ik[34], road4.N_ik[35]]
        road4_Q_ik_back = [road4.Q_ik[33], road4.Q_ik[34], road4.Q_ik[35]]
        road4.Q_ik[33] = int(road4_Q_ik_back[0] * 0.1)
        road4.Q_ik[34] = int(road4_Q_ik_back[1] * 0.1)
        road4.Q_ik[35] = int(road4_Q_ik_back[2] * 0.1)

        road4.N_ik[33] = int(road4_N_ik_back[0] * 0.1)
        road4.N_ik[34] = int(road4_N_ik_back[1] * 0.1)
        road4.N_ik[35] = int(road4_N_ik_back[2] * 0.1)

    if step == 60 * 60 / 5:  # 如果在30分钟
        road4.Q_ik[33] = int(road4_Q_ik_back[0])
        road4.Q_ik[34] = int(road4_Q_ik_back[1])
        road4.Q_ik[35] = int(road4_Q_ik_back[2])

        road4.N_ik[33] = int(road4_N_ik_back[0])
        road4.N_ik[34] = int(road4_N_ik_back[1])
        road4.N_ik[35] = int(road4_N_ik_back[2])


x = np.arange(1, epoch_num + 1) * 5
y = np.array(jam_cars_num_all)
plt.title("Matplotlib demo")
plt.xlabel("time step")
plt.ylabel("number of jam_cells")
plt.plot(x, y)
plt.show()
