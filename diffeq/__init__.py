import numpy as np
import scipy.linalg as la


def solve_lin_ord2_h_im(a, b, c, x, y, y_prime):
    """
    Solve homogeneous 2nd order linear differential equation with imaginary roots.

    :param a: coefficient of y''
    :param b: coefficient of y'
    :param c: coefficient of y
    :param x: initial value of x
    :param y: initial value of y
    :param y_prime: initial value of y'
    :return: a tuple containing the constants c1 and c2
    """
    alpha = -b / (2.0 * a)
    beta = np.sqrt(4.0 * a * c - b * b) / (2.0 * a)
    print(f"λ_1 = {alpha} + {beta}i λ_2 = {alpha} - {beta}i")
    t = np.asmatrix([
        [np.cos(beta * x), np.sin(beta * x)]
    ])
    d = t * np.asmatrix([
        [alpha, beta],
        [-beta, alpha]
    ])
    m = np.exp(alpha * x) * np.asmatrix([
        t.A[0],
        d.A[0]
    ])
    s = np.asmatrix([
        [y],
        [y_prime]
    ])
    # todo account for negative values of alpha and beta by changing the +/- to fit
    print(f"General solution:\ny = exp({alpha}x)(c_1 ({alpha}cos({beta}x) - {beta}sin({beta}x)) + c_2 ({beta}cos({beta}x) + {alpha}sin({beta}x)))")
    solution = tuple(la.lu_solve(la.lu_factor(m), s).flat)
    print(f"c_1 = {solution[0]} c_2 = {solution[1]}")
    return solution
