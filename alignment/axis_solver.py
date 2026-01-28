import numpy as np
import itertools


def solve_axis_mapping(sfm_cam, gps_cam):

    sfm_vec = sfm_cam[-1] - sfm_cam[0]
    gps_vec = gps_cam[-1] - gps_cam[0]

    axes = [0, 1, 2]
    signs = [-1, 1]

    best_M = None
    best_err = 1e9

    for perm in itertools.permutations(axes):
        for sx in signs:
            for sy in signs:
                for sz in signs:

                    M = np.zeros((3, 3))

                    M[0, perm[0]] = sx
                    M[1, perm[1]] = sy
                    M[2, perm[2]] = sz

                    mapped = M @ sfm_vec

                    mapped_n = mapped / np.linalg.norm(mapped)
                    gps_n = gps_vec / np.linalg.norm(gps_vec)

                    err = np.linalg.norm(mapped_n - gps_n)

                    if err < best_err:
                        best_err = err
                        best_M = M

    print("\n===== AXIS AUTO SOLVER =====")
    print("Best mapping matrix:\n", best_M)
    print("Direction error:", best_err)

    return best_M
