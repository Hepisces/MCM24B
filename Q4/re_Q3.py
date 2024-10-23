import numpy as np
from tqdm import trange
from bayes import p_correct as p_correct

def calculate(way, price, N, iter, p,sigma, pr=False):
    N0 = N
    p0=p

    results = np.zeros(iter)
    for time in range(iter):
        final_cost = 0
        p=p_correct(1-p0,sigma)
        # 生成八个零件的随机质量序列
        lj1 = sequence(N, p)
        lj2 = sequence(N, p)
        lj3 = sequence(N, p)
        lj4 = sequence(N, p)
        lj5 = sequence(N, p)
        lj6 = sequence(N, p)
        lj7 = sequence(N, p)
        lj8 = sequence(N, p)

        # 计算零件的总成本(包括采购和检测)
        lj_price = 0
        for i in range(1, 9):
            name = f"lj{i}"
            lj_price += price[name] + price[f"j_{name}"]
        final_cost += lj_price * N
        print(f"N:{N}") if pr else None
        # 检查这8个零件，并做对齐和库存

        # 对齐BC1
        N_bc1 = min(np.sum(lj1), np.sum(lj2), np.sum(lj3))
        # lj1 = [1] * N_bc1
        # lj2 = [1] * N_bc1
        # lj3 = [1] * N_bc1
        for name in ["lj1", "lj2", "lj3"]:
            bc1_price = price[name] + price[f"j_{name}"]
            final_cost -= bc1_price * (N0 - N_bc1)
        print(f"N_bc1:{N_bc1}") if pr else None

        # 对齐BC2
        N_bc2 = min(np.sum(lj4), np.sum(lj5), np.sum(lj6))
        # lj4 = [1] * N_bc2
        # lj5 = [1] * N_bc2
        # lj6 = [1] * N_bc2
        for name in ["lj4", "lj5", "lj6"]:
            bc2_price = price[name] + price[f"j_{name}"]
            final_cost -= bc2_price * (N0 - N_bc2)

        print(f"N_bc2:{N_bc2}") if pr else None
        # 对齐BC3
        N_bc3 = min(np.sum(lj7), np.sum(lj8))
        # lj7 = [1] * N_bc3
        # lj8 = [1] * N_bc3
        for name in ["lj7", "lj8"]:
            bc3_price = price[name] + price[f"j_{name}"]
            final_cost -= bc3_price * (N0 - N_bc3)
        print(f"N_bc3:{N_bc3}") if pr else None

        nums = 0
        while nums <= 3:
            p=p_correct(1-p0,sigma)
            print(f"nums:{nums}") if pr else None

            # 合成3个半成品，生成半成品的随机序列，并对齐
            if nums == 0:
                bc1 = sequence(N_bc1, p)
                bc2 = sequence(N_bc2, p)
                bc3 = sequence(N_bc3, p)

            bc_min = min(N_bc1, N_bc2, N_bc3)

            # 计算半成品组装成本及库存
            if nums == 0:
                final_cost += price["bc"] * 3 * bc_min
            else:
                final_cost += (
                    (3 - way["bc1"] - way["bc2"] - way["bc3"]) * price["bc"] * bc_min
                )

            if way["bc1"]:
                final_cost += N_bc1 * price["j_bc"]
                bc1 = [1] * np.sum(bc1)
                N_bc1 = np.sum(bc1)

            if way["bc2"]:
                final_cost += N_bc2 * price["j_bc"]
                bc2 = [1] * np.sum(bc2)
                N_bc2 = np.sum(bc2)

            if way["bc3"]:
                final_cost += N_bc3 * price["j_bc"]
                bc3 = [1] * np.sum(bc3)
                N_bc3 = np.sum(bc3)

            if np.all(bc1 == 1) and np.all(bc2 == 1) and np.all(bc3 == 1):
                break

            if pr:
                print(f"N_bc1:{N_bc1}")
                print(f"N_bc2:{N_bc2}")
                print(f"N_bc3:{N_bc3}")
            nums += 1

        # 成品检测与否
        N_bc_min = min(N_bc1, N_bc2, N_bc3)
        bc1 = bc1[:N_bc_min]
        bc2 = bc2[:N_bc_min]
        bc3 = bc3[:N_bc_min]
        cp = sequence(N_bc_min, p) * bc1 * bc2 * bc3
        final_cost -= price["sale"] * np.sum(cp)

        print(f"cp:{np.sum(cp)}") if pr else None

        if way["cp"]:
            final_cost += price["j_cp"] * cp.shape[0]
        else:
            final_cost += price["diao"] * (cp.shape[0] - np.sum(cp))

        if way["chai"]:
            final_cost += price["c_cp"] * (cp.shape[0] - np.sum(cp))
            idx = np.where(cp == 0)[0]
            print(f"idx:{len(idx)}") if pr else None
            bc1 = np.array([bc1[i] for i in idx])
            bc2 = np.array([bc2[i] for i in idx])
            bc3 = np.array([bc3[i] for i in idx])
            print(f"bc1:{len(bc1)}") if pr else None

        nums = 0
        while nums <= 3 and way["chai"]:
            print(f"nums:{nums}") if pr else None

            # bc_min = min(N_bc1, N_bc2, N_bc3)
            # bc1 = sequence(N_bc1, p) if (not way["re_bc1"] and nums != 0) else bc1
            # bc2 = sequence(N_bc2, p) if (not way["re_bc2"] and nums != 0) else bc2
            # bc3 = sequence(N_bc3, p) if (not way["re_bc3"] and nums != 0) else bc3

            print(f"bc1:{len(bc1)}") if pr else None

            # 计算半成品组装成本及库存
            if nums == 0:
                final_cost += price["bc"] * 3 * bc_min
            else:
                final_cost += (
                    (3 - way["re_bc1"] - way["re_bc2"] - way["re_bc3"])
                    * price["bc"]
                    * bc_min
                )

            if way["re_bc1"]:
                final_cost += N_bc1 * price["j_bc"]
                bc1 = np.array([1] * np.sum(bc1))
                N_bc1 = np.sum(bc1)

            if way["re_bc2"]:
                final_cost += N_bc2 * price["j_bc"]
                bc2 = np.array([1] * np.sum(bc2))
                N_bc2 = np.sum(bc2)

            if way["re_bc3"]:
                final_cost += N_bc3 * price["j_bc"]
                bc3 = np.array([1] * np.sum(bc3))
                N_bc3 = np.sum(bc3)

            bc_min = min(bc1.shape[0], bc2.shape[0], bc3.shape[0])
            bc1 = bc1[:bc_min]
            bc2 = bc2[:bc_min]
            bc3 = bc3[:bc_min]

            if np.all(bc1 == 1) and np.all(bc2 == 1) and np.all(bc3 == 1):
                break

            if pr:
                print(f"N_bc1:{bc1.shape[0]}")
                print(f"N_bc2:{bc2.shape[0]}")
                print(f"N_bc3:{bc3.shape[0]}")
            nums += 1

        results[time] = -final_cost

    return results / N0


def sequence(num, p):
    return np.random.binomial(1, 1 - p, num)


# if __name__ == "__main__":
#     price = {
#         "lj1": 2,
#         "j_lj1": 1,
#         "lj2": 8,
#         "j_lj2": 1,
#         "lj3": 12,
#         "j_lj3": 1,
#         "lj4": 2,
#         "j_lj4": 1,
#         "lj5": 8,
#         "j_lj5": 1,
#         "lj6": 12,
#         "j_lj6": 2,
#         "lj7": 8,
#         "j_lj7": 1,
#         "lj8": 12,
#         "j_lj8": 2,
#         "bc": 8,
#         "j_bc": 4,
#         "c_bc": 6,
#         "cp": 8,
#         "j_cp": 6,
#         "c_cp": 10,
#         "sale": 200,
#         "diao": 40,
#     }
# check = ["bc1", "bc2", "bc3", "cp", "chai", "re_bc1", "re_bc2", "re_bc3"]
# tf = [1, 1, 1, 1, 0, 0, 0, 0]

# p = 0.1
# N = 1000
# iter = 1000
# pr = False
# sigma=0.5

# way = dict(zip(check, tf))
# results = calculate(way, price, N, iter, p,sigma, pr)
# print(results.mean())
# print(results.std())

# from matplotlib.pyplot import hist
# import matplotlib.pyplot as plt

# hist(results, bins=100)
# plt.show()
