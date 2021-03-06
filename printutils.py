import numpy as np
from typing import Union, List


def plurality_conjugate(x, singular, plural=None):
    if plural is not None:
        return singular if x == 1 else plural
    else:
        return singular if x == 1 else singular + "s"


def oxford_list(lst, fmt=None, sep=", ", oxford_comma=True, comma_on_2=False, final_conjunction="and"):
    ret = ""
    for i in range(len(lst)):
        item = lst[i]
        if fmt:
            if hasattr(item, "__iter__"):
                ret += fmt.format(*item)
            else:
                ret += fmt.format(item)
        else:
            ret += item
        remaining = len(lst) - 1 - i
        if remaining > 1:
            ret += sep
        elif remaining == 1:
            if oxford_comma:
                if len(lst) == 2 and not comma_on_2:
                    ret += " "
                else:
                    ret += sep
            if final_conjunction:
                ret += f"{final_conjunction} "
    print(ret)
    return ret


def latex_format_vec(vec, precision=3):
    ret = ""
    if len(vec) == 2:
        x, y = vec
        z = 0
    else:
        x, y, z = vec
    if x != 0:
        ret += str(round(x, ndigits=precision)) + "\\hat{i}"
    if y != 0:
        if y > 0:
            ret += "+"
        ret += str(round(y, ndigits=precision)) + "\\hat{j}"
    if z != 0:
        if z > 0:
            ret += "+"
        ret += str(round(z, ndigits=precision)) + "\\hat{k}"
    if ret == "":
        return "0"
    return ret.removeprefix("+")


def latex_format_matrix(matrix: Union[List[List[str]], List[List[float]], np.ndarray], force_vertical=False, precision=3):
    if type(matrix) is list:
        if type(matrix[0][0]) is str:
            return "\\begin{bmatrix}\n" + "\\\\\n".join(map(" & ".join, matrix[:])) + "\n\\end{bmatrix}"
        matrix = np.asarray(matrix)

    dimensionality = len(matrix.shape)
    if dimensionality > 2:
        raise ValueError("Argument is a tensor, not a matrix")
    if dimensionality == 0:
        raise ValueError("Empty array")

    if force_vertical and dimensionality == 1:
        matrix = matrix.reshape((len(matrix), 1))

    return "\\begin{bmatrix}\n" + "\\\\\n".join(map(" & ".join, map(lambda row: map("{:.3f}".format, row), matrix[:]))) + "\n\\end{bmatrix}"
