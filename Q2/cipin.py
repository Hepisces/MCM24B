def cpcipin(p0, p1, p2, p1test=False, p2test=False):
    """
    计算成品的次品率\\
    p0: 零件均为正品时的成品次品率\\
    p1: 零件1的次品率\\
    p2: 零件2的次品率\\
    p1test: 是否检测零件1 (True表示检测)\\
    p2test: 是否检测零件2 (True表示检测)\\
    return: 成品的次品率
    """
    
    # 如果不检测任何零件，计算完整次品率
    if not p1test and not p2test:
        p = p0 * (1 - p1) * (1 - p2) + p1 * (1 - p2) + p2 * (1 - p1) + p1 * p2
    
    # 如果检测了零件1，但未检测零件2
    if p1test and not p2test:
        p = p0 * (1 - p2) + p2
    
    # 如果检测了零件2，但未检测零件1
    if not p1test and p2test:
        p = p0 * (1 - p1) + p1
    
    # 如果检测了零件1和零件2
    if p1test and p2test:
        p = p0
    
    return p


if __name__ == '__main__':
    p = cpcipin(0.1, 0.1, 0.1, p1test=True, p2test=False)
    print(f"{p:.4f}")
