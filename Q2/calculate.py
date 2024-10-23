import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist


def calculate(N, way, p0, p1, p2, price, n=10000):
    """
    way:一个字典，key是A-F，对应各个检测流程，value是是否要检测\\
    p0:零件均为正品时的成品次品率\\
    p1:零件1是次品的概率\\
    p2:零件2是次品的概率\\
    price:一个字典，key:[lj1,lj2,j_lj1,j_lj2,cp,j_cp,sale]，value是对应费用
    """

    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer")
    if not isinstance(way, dict) or any(key not in "ABCDEF" for key in way):
        raise ValueError("way must be a dictionary with keys A-F")
    if not isinstance(price, dict) or any(
        key not in price
        for key in [
            "lj1",
            "lj2",
            "j_lj1",
            "j_lj2",
            "cp",
            "j_cp",
            "sale",
            "diaohuan",
            "chaijie",
        ]
    ):
        raise ValueError("price must be a dictionary with specific keys")

    N0 = N

    results = np.zeros(n)
    for time in range(n):
        lj1 = sequence(N0, 1 - p1)
        lj2 = sequence(N0, 1 - p2)

        final_cost = N0 * (price["lj1"] + price["lj2"])

        # 装配前检查

        if way["A"]:
            # 如果流程A检查，那么所有的零件1都是正品,
            # 这时候的花费应该是零件1的检测费用
            final_cost += cost(lj1, [price["j_lj1"]])
            lj1 = np.array([1] * np.sum(lj1))
        if way["B"]:
            # 如果流程B检查，那么所有的零件2都是正品,
            # 这时候的花费应该是零件2的检测费用
            final_cost += cost(lj2, [price["j_lj2"]])
            lj2 = np.array([1] * np.sum(lj2))

        N = min(len(lj1), len(lj2))
        cha = len(lj1) - len(lj2)
        final_cost -= price["lj1"] * cha if cha > 0 else price["lj2"] * (-cha)

        lj1 = lj1[:N]
        lj2 = lj2[:N]

        p_cipin = p0
        cp = sequence(N, 1 - p_cipin)
        cp = np.array([cp[i] if lj1[i] * lj2[i] == 1 else 0 for i in range(len(cp))])
        # 计算成品的组装费用和销售费用
        final_cost -= price["sale"] * np.sum(cp)  # 销售费用，只算正品
        final_cost += price["cp"] * len(cp)  # 成品组装费用

        nums = 0
        p_cipin = p0
        # 成品检查
        while np.any(lj1 * lj2 == 0) and nums<2:
            # print(nums)

            # 现在制作成品,针对不同的检测流程，次品率是不一样的,这里需要计算
            # from cipin import cpcipin

            # p_cipin = (
            #     cpcipin(p0, p1, p2, p1test=way["A"], p2test=way["B"])
            #     if nums == 0
            #     else cpcipin(p0, p1, p2, p1test=way["E"], p2test=way["F"])
            # )
            # 生成成品的次品判别序列
            if nums > 0:
                cp = sequence(N, 1 - p_cipin)
                cp = np.array(
                    [cp[i] if lj1[i] * lj2[i] == 1 else 0 for i in range(len(cp))]
                )
                # 组装费用
                final_cost += price["cp"] * len(cp)
            # 再加成品的检测费用(如果有)
            if way["C"]:
                final_cost += len(cp)*price["j_cp"]
                if not way["D"]:
                    break
                cp_cipin_idxs = np.where(cp == 0)[0].tolist()
            else:
                # 如果没有检查，意味着有次品流入市场，必然支付调换费用
                diao_num = np.where(cp == 0)[0].shape[0]
                final_cost += price["diaohuan"] * diao_num
                cp_cipin_idxs = np.where(cp == 0)[0].tolist()
                if not way["D"]:
                    break

            # 拆解后检查
            # 两种情况，一种是成品在组装时已经检查，一种是没有检查直接流入市场;但是殊途同归
            if way["D"]:
                # 如果拆解，就支付拆解费用，进而讨论零件的流向
                lj1 = np.array([lj1[i] for i in cp_cipin_idxs])
                lj2 = np.array([lj2[i] for i in cp_cipin_idxs])
                final_cost += price["chaijie"] * len(cp_cipin_idxs)

            if way["E"] and not way["A"]:
                # 这时候的花费应该是零件1的检测费用
                final_cost += cost(lj1, [price["j_lj1"]])
                lj1 = np.array([1] * np.sum(lj1))

            if way["F"] and not way["B"]:
                # 这时候的花费应该是零件2的检测费用
                final_cost += cost(lj2, [price["j_lj2"]])
                lj2 = np.array([1] * np.sum(lj2))

            N = min(len(lj1), len(lj2))
            cha = len(lj1) - len(lj2)
            final_cost -= price["lj1"] * cha if cha > 0 else price["lj2"] * (-cha)

            lj1 = lj1[:N]
            lj2 = lj2[:N]

            # if not way["D"] and not way["E"] and not way["F"]:
            #     break
            nums += 1

        results[time] = -final_cost

        # 输出当前的成本，观察变化
        # print(f"Iteration {time}:  final_cost: {final_cost}") if time % (N0/10) == 0 else None

        lj1, lj2, cp = None, None, None
        final_cost = None

    # 最后返回结果
    return results/N0


def sequence(num, prob):
    """
    num:总数
    prob:概率
    """
    return np.random.binomial(1, prob, num)


def cost(lists, price):
    """
    lists:待计算的序列\\
    """
    lists = np.array(lists)

    # if type == "lj":
    #     # 如果是零件，那么费用是检测费用
    #     jiance = lists.shape[0] * price[0]
    #     return jiance

    # if type == "cp":
    #     # 如果是成品，那么费用有两部分，一部分是检测费用一部分是销售价格
    #     jiance = lists.shape[0] * price[0]
    #     xiaoshou = np.sum(lists) * price[1]
    return lists.shape[0] * price[0]


# if __name__ == "__main__":
#     # 测试代码
#     N = 10000
#     way = {"A": 1, "B": 1, "C": 1, "D": 0, "E": 0, "F": 0}
#     p0 = 0.1
#     p1 = 0.1
#     p2 = 0.1
#     price = {
#         "lj1": 4,
#         "lj2": 18,
#         "j_lj1": 2,
#         "j_lj2": 3,
#         "cp": 6,
#         "j_cp": 3,
#         "sale": 56,
#         "diaohuan": 6,
#         "chaijie": 5,
#     }
#     result = calculate(N, way, p0, p1, p2, price, n=10000)
#     print(result.mean())
#     print(result.std())
#     # print(result[2])

#     hist(result[2])
#     plt.show()
