from statics import *
import sys

A_pos = [0, 0]
B_pos = [3, 3]
C_pos = [6, 3]
D_pos = [9, 3]
E_pos = [6, -2]
F_pos = [3, 0]

A = Joint(A_pos, "A", [0, -5])
B = Joint(B_pos, "B", [0, -4])
C = Joint(C_pos, "C")
D = Joint(D_pos, "D")
E = Joint(E_pos, "E")
F = Joint(F_pos, "F")

A.connect(B, F, E)
B.connect(A, F, C)
C.connect(B, F, E, D)
D.connect(C, E)
E.connect(A, C, D)
F.connect(A, B, C)

if len(sys.argv) != 2:
    print("1 command line argument required: output file")

with open(sys.argv[1], "w") as file:
    solve_truss_2d([D, E, C, B, A, F], D, E, [0, 1], force_unit="kN", latex_file=file)
