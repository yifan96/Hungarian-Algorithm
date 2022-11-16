"""
Author: Zhongqiang (Richard) Ren
All Rights Reserved.
ABOUT: Entrypoint to the code.
Oeffentlich fuer: RSS22
"""

import context
import time
import numpy as np
import random
import cbss_msmp
import cbss_mcpf

import common as cm


def run_CBSS_MSMP():
    """
    fully anonymous case, no assignment constraints.
    """
    print("------run_CBSS_MSMP------")
    ny = 100
    nx = 100
    grids = np.zeros((ny, nx))
    grids[5, 3:7] = 1  # obstacles
    print(grids)

    starts = [11, 22, 33, 88, 99, 444, 443, 232, 43, 34]
    targets = [40, 38, 27, 66, 72, 81, 83, 123, 343, 23, 555, 342, 456, 778, 997, 331, 1, 233, 400, 329]
    # dests = [19,28,37,46,69                       ,223,654,232,31,76]
    dests = [11, 22, 33, 88, 99, 444, 443, 232, 43, 34]

    configs = dict()
    configs["problem_str"] = "msmp"
    configs["mtsp_fea_check"] = 1
    configs["mtsp_atLeastOnce"] = 1
    # this determines whether the k-best TSP step will visit each node for at least once or exact once.
    configs["tsp_exe"] = "./pytspbridge/tsp_solver/LKH-2.0.9/LKH"
    configs["time_limit"] = 60
    configs["eps"] = 0.0
    res_dict = cbss_msmp.RunCbssMSMP(grids, starts, targets, dests, configs)

    print(res_dict)

    return


def run_CBSS_MCPF():
    """
    With assignment constraints.
    """
    print("------run_CBSS_MCPF------")
    ny = 10
    nx = 10
    grids = np.zeros((ny, nx))
    grids[5, 3:7] = 1  # obstacles

    starts = [11, 22, 33, 88, 99]
    targets = [72, 81, 83, 40, 38, 27, 66]
    dests = [46, 69, 19, 28, 37]
    # heterogenous agents (f_A in paper CBSS_MCPF)
    ac_dict = dict()
    ri = 0
    for k in targets:
        ac_dict[k] = set([ri, ri + 1])
        ri += 1
        if ri >= len(starts) - 1:
            break
    ri = 0
    for k in dests:
        ac_dict[k] = set([ri])
        ri += 1
    print("Assignment constraints : ", ac_dict)

    configs = dict()
    configs["problem_str"] = "msmp"
    configs["mtsp_fea_check"] = 1
    configs["mtsp_atLeastOnce"] = 1
    # this determines whether the k-best TSP step will visit each node for at least once or exact once.
    configs["tsp_exe"] = "./pytspbridge/tsp_solver/LKH-2.0.9/LKH"
    configs["time_limit"] = 60
    configs["eps"] = 0.0

    res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, ac_dict, configs)

    print(res_dict)

    return


if __name__ == '__main__':
    print("begin of main")

    run_CBSS_MSMP()

    # run_CBSS_MCPF()

    print("end of main")