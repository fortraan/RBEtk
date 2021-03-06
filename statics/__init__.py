import numpy as np
from dataclasses import dataclass, field
from typing import List, Union
import printutils
import sys

# this code is messy and needs to be refactored.
# todo clean this up

np.set_printoptions(suppress=True)


def solve_force_matrix(dirs: Union[list, np.ndarray], known: Union[List[list], list, np.ndarray]):
    # allow for list of vectors or a matrix of column vectors
    if type(dirs) is list:
        dirs = np.asarray(dirs).transpose()
    if type(known) is List[list]:
        known = np.asarray(known).transpose()
    elif type(known) is list:
        known = np.asarray(known).reshape((len(known), 1))
    known = -known.sum(axis=1)
    try:
        ret = np.linalg.solve(dirs, known)
    except np.linalg.LinAlgError:
        ret, _, _, _ = np.linalg.lstsq(dirs, known, rcond=None)
    print(f"{dirs}{ret} =\n{known.reshape((len(known), 1))}")
    return ret


class Joint:
    def __init__(self, position, name: str, *loads: list):
        self.pos = np.asarray(position)
        self.name = name
        self.loading = np.zeros((len(position), 1), dtype=np.float64)
        self.reaction_loading = self.loading.copy()
        self.num_loads = 0
        for load in loads:
            self.add_load(load)
        self.connections = list()
        self.connected_joints = list()
        self.has_reactions = False

    def add_load(self, load: list):
        self.loading += np.transpose([load])
        self.num_loads = self.num_loads + 1

    def add_reaction(self, load: list):
        self.has_reactions = True
        self.reaction_loading += np.transpose([load])

    def connect(self, *others):
        new_joints = list(filter(lambda j: j not in self.connections, others))
        self.connections.extend([Connection(self, joint) for joint in new_joints])
        self.connected_joints.extend(new_joints)

    def get_known_forces(self):
        return [connection for connection in self.connections if connection.solved]

    def get_unknown_forces(self):
        return [connection for connection in self.connections if not connection.solved]

    def num_connections(self):
        return len(self.connections)

    def num_known_forces(self):
        return len(self.get_known_forces())

    def num_unknown_forces(self):
        return self.num_connections() - self.num_known_forces()

    def determinate(self):
        return self.num_unknown_forces() < 3 and self.num_known_forces() > 0

    def symbolic_knowns(self):
        retX, retY = dict(), dict()
        if self.has_reactions:
            retX[self.name + "_x"] = self.reaction_loading[0, 0]
            retY[self.name + "_y"] = self.reaction_loading[1, 0]
        if self.num_loads > 0:
            retX["L_x"] = self.loading[0, 0]
            retY["L_y"] = self.loading[1, 0]
        for connection in self.connections:
            if connection.solved:
                retX["T_{" + connection.get_name() + "|x}"] = round(connection.load_vec[0], ndigits=3)
                retY["T_{" + connection.get_name() + "|y}"] = round(connection.load_vec[1], ndigits=3)
        return retX, retY

    def symbolic_unknowns(self):
        dir_x, dir_y, names = list(), list(), list()
        for connection in self.connections:
            if not connection.solved:
                base_name = "\\left(\\widehat{" + connection.get_name() + "}\\right)"
                dir_x.append(base_name + "_{x}")
                dir_y.append(base_name + "_{y}")
                names.append(["T_{" + connection.get_name() + "}"])
        return dir_x, dir_y, names


@dataclass
class Connection:
    start: Joint
    end: Joint
    dir: np.ndarray = field(default=None, init=False)
    solved: bool = False
    load: float = field(default=None, init=False)
    load_vec: np.ndarray = field(default=None, init=False)

    def __post_init__(self):
        self.dir = np.asarray(self.end.pos - self.start.pos, dtype=np.float64)
        self.dir /= np.linalg.norm(self.dir)

    def set_load(self, load, share_data=True):
        self.solved = True
        self.load = load
        self.load_vec = self.dir * load
        if share_data:
            for connection in self.end.connections:
                if connection.end == self.start:
                    connection.set_load(load, False)

    def same_endpoints(self, other):
        return self.start == other.end and self.end == other.start

    def get_name(self):
        return self.start.name + self.end.name


def make_member(a: Joint, b: Joint):
    a.connect(b)
    b.connect(a)


def solve_truss_2d(joints: List[Joint], pin_support: Joint, roller_support: Joint, roller_axis: list, auto_order=True, length_unit="m", force_unit="N", latex_file=sys.stdout):
    all_joints = {*joints, pin_support, roller_support}
    load_str = ""
    moment_str = ""
    overall_load = np.zeros((2, 1))
    overall_moment = 0
    # calculate overall load and moment due to external forces
    # these are used to calculate reaction forces
    latex_file.write("\\begin{align*}\n")
    for joint in sorted(all_joints, key=lambda j: j.name):
        latex_file.write("\\vec{" + joint.name + "} &= " + printutils.latex_format_vec(joint.pos) + " &")
        moment_arm = joint.pos - pin_support.pos
        overall_moment += np.cross(moment_arm, joint.loading[:, 0])
        overall_load += joint.loading
        if joint.num_loads > 0:
            if load_str:
                v = printutils.latex_format_vec(joint.loading[:, 0])
                if not v.startswith("-"):
                    load_str += "+" + v
                else:
                    load_str += v
            else:
                load_str = printutils.latex_format_vec(joint.loading[:, 0])

            v = f"\\overrightarrow{{{pin_support.name + joint.name}}}\\times\\left(" + printutils.latex_format_vec(joint.loading[:, 0]) + "\\right)"
            if moment_str:
                moment_str += "+" + v
            else:
                moment_str += v

    latex_file.write("\\end{align*}\n\\begin{gather*}\n\\sum_n F_n = 0 =")
    latex_file.write(load_str)
    latex_file.write(f"+{pin_support.name}_x +{pin_support.name}_y +{roller_support.name}_x +{roller_support.name}_y" +
                     "\\\\\n\\sum_n M_{n|" + pin_support.name + "} = 0 =" +
                     f"\\overrightarrow{{{pin_support.name + roller_support.name}}}\\times\\left({roller_support.name}_x \\hat{{i}} + {roller_support.name}_y \\hat{{j}}\\right) + {moment_str}")
    latex_file.write(f"\\\\\n{roller_support.name}_x \\hat{{i}} + {roller_support.name}_y \\hat{{j}} \\propto ")
    latex_file.write(printutils.latex_format_vec(roller_axis))
    latex_file.write("\\end{gather*}\n")
    print(f"Overall load on the structure: {overall_load[:, 0]} {force_unit}\nMoment due to this load: {overall_moment} {force_unit} {length_unit}")
    roller_axis_vec = np.asarray(roller_axis) / np.linalg.norm(roller_axis)
    # set up a matrix to solve for reaction forces in the first 2 rows and moment in the third
    force_matrix = np.asarray([
        [1, 0, roller_axis_vec[0]],
        [0, 1, roller_axis_vec[1]],
        [0, 0, np.cross(roller_support.pos - pin_support.pos, roller_axis_vec)]
    ])
    resultant = np.asarray([
        [overall_load[0, 0]],
        [overall_load[1, 0]],
        [overall_moment]
    ])
    reaction_force_coefs = solve_force_matrix(force_matrix, resultant)
    pin_reaction = reaction_force_coefs[:-1]
    roller_reaction = roller_axis_vec * reaction_force_coefs[-1]
    latex_file.write(f"\\[\\begin{{bmatrix}}\n"
                     f"1 & 0 & {roller_axis_vec[0]}\\\\\n"
                     f"0 & 1 & {roller_axis_vec[1]}\\\\\n"
                     f"0 & 0 & \\overrightarrow{{{pin_support.name + roller_support.name}}}\\times\\left(" +
                     printutils.latex_format_vec(roller_axis_vec) +
                     "\\right)\n"
                     "\\end{bmatrix}"
                     "\\begin{bmatrix}\n"
                     f"{pin_support.name}_x\\\\\n"
                     f"{pin_support.name}_y\\\\\n"
                     f"{roller_support.name}_y\n"
                     "\\end{bmatrix} = -\n" +
                     printutils.latex_format_matrix(resultant, force_vertical=True) +
                     "\\]\n"
                     "\\begin{align*}\n"
                     f"{pin_support.name}_x &= {pin_reaction[0]} & "
                     f"{pin_support.name}_y &= {pin_reaction[1]} & "
                     f"{roller_support.name}_y &= {roller_reaction[1]} \n"
                     "\\end{align*}\n")
    print(f"Reaction force from the pin support: {pin_reaction} {force_unit}\n"
          f"Reaction force from the roller support: {roller_reaction} {force_unit}")
    # reaction forces behave like external loads
    pin_support.add_reaction(pin_reaction)
    roller_support.add_reaction(roller_reaction)

    if auto_order:
        # start with the pin support, since we know the most about it
        __solve_joint_2d(pin_support, latex_file=latex_file)

        unsolved_joints = list(all_joints)
        unsolved_joints.remove(pin_support)

        # traverse through the unsolved joints
        # next joint is always connected to the previous one
        while len(unsolved_joints) > 0:
            connected_joints = list(filter(Joint.determinate, unsolved_joints))
            connected_joints.sort(key=Joint.num_known_forces)
            connected_joints.reverse()
            if len(connected_joints) == 0:
                print("Solve failed. The following joints remain indeterminate: " + ", ".join(unsolved_joints))
                raise np.linalg.LinAlgError("No more joints are determinate")
            for to_solve in connected_joints:
                try:
                    __solve_joint_2d(to_solve, latex_file=latex_file)
                    unsolved_joints.remove(to_solve)
                except np.linalg.LinAlgError:
                    print(f"Failed to solve joint {to_solve.name}, trying a different joint")
                    continue
    else:
        for joint in joints:
            __solve_joint_2d(joint, latex_file=latex_file)

    latex_file.write("\\\\\n\\\\\n")
    print("Solution:")
    print(f"Pin Reaction: {pin_reaction} {force_unit}\nRoller Reaction: {roller_reaction} {force_unit}")
    member_force_dict = dict()
    unique_connections = []
    for joint in all_joints:
        for connection in joint.connections:
            if not any(map(connection.same_endpoints, unique_connections)):
                unique_connections.append(connection)
                load_type = "tensile" if connection.load > 0 else "compressive"
                latex_file.write(f"Member {connection.start.name}{connection.end.name} is under a {load_type} load of {abs(connection.load):.3f} {force_unit}.\\\\\n")
                member_force_dict[connection.start.name + connection.end.name] = connection.load
                member_force_dict[connection.end.name + connection.start.name] = connection.load
    return member_force_dict


def __solve_joint_2d(joint: Joint, latex_file=sys.stdout):
    print(f"Solving joint {joint.name}...")
    # find known and unknown connection forces
    known_connections = []
    unknown_connections = []
    unknown_connection_dirs = []
    known_force_resultant = joint.loading[:, 0] + joint.reaction_loading[:, 0]
    for connection in joint.connections:
        if not connection.solved:
            unknown_connections.append(connection)
            unknown_connection_dirs.append(connection.dir)
        else:
            known_connections.append(connection)
            known_force_resultant = np.add(known_force_resultant, connection.load_vec)

    num_known = len(known_connections)
    num_unknown = len(unknown_connections)
    joint_noun = printutils.plurality_conjugate(joint.num_connections(), "joint")
    known_verb = printutils.plurality_conjugate(num_known, "is", "are")
    unknown_verb = printutils.plurality_conjugate(num_unknown, "is", "are")
    known_names = list(map(Connection.get_name, known_connections))
    unknown_names = list(map(Connection.get_name, unknown_connections))
    print(unknown_names)
    print(f"Joint {joint.name} is connected to {joint.num_connections()} other {joint_noun}. {num_known} of "
          f"those connections {known_verb} solved and {num_unknown} {unknown_verb} unsolved. ", file=latex_file, end="")
    if num_unknown > 0:
        latex_file.write(f"The unknown connections are {printutils.oxford_list(unknown_names)}. ")
    if num_known > 0:
        latex_file.write(f"The known connections {known_verb} {printutils.oxford_list(known_names)}. ")
    elif num_known == 0:
        latex_file.write("Although no connections have been solved, this joint has ")
        if joint.num_loads > 0:
            latex_file.write(f"{joint.num_loads} known external loads")
        if joint.has_reactions:
            if joint.num_loads > 0:
                latex_file.write(" and ")
            latex_file.write("known reaction forces")
        latex_file.write(".")
    latex_file.write("\n")
    print("Unknown: [" + ", ".join(unknown_names)
          + "] Known: [" + ", ".join(known_names) + "]")

    if len(unknown_connections) > 2:
        raise np.linalg.LinAlgError(f"Joint {joint.name} is indeterminate")
    if len(unknown_connections) == 0:
        latex_file.write("All forces on this joint have already been solved.\n")
        return

    resultant_symbols_x, resultant_symbols_y = joint.symbolic_knowns()
    resultant_name_x = " + ".join(resultant_symbols_x.keys())
    resultant_name_y = " + ".join(resultant_symbols_y.keys())
    resultant_values_x = printutils.oxford_list(list(resultant_symbols_x.values()), fmt="{0:+.3f}", sep=" ", final_conjunction=None).removeprefix("+")
    resultant_values_y = printutils.oxford_list(list(resultant_symbols_y.values()), fmt="{0:+.3f}", sep=" ", final_conjunction=None).removeprefix("+")
    resultant_x = sum(resultant_symbols_x.values())
    resultant_y = sum(resultant_symbols_y.values())

    dir_x, dir_y, dir_names = joint.symbolic_unknowns()

    latex_file.write(f"\\[\n")
    latex_file.write(printutils.latex_format_matrix([dir_x, dir_y], force_vertical=True) + "\n")
    latex_file.write(printutils.latex_format_matrix(dir_names, force_vertical=True) + "\n")
    latex_file.write(" = -")
    latex_file.write(printutils.latex_format_matrix([[resultant_name_x], [resultant_name_y]], force_vertical=True))
    latex_file.write("\\]\\begin{align*}\n")
    latex_file.write(" & ".join([f"\\widehat{{{c.get_name()}}} &= {printutils.latex_format_vec(c.dir)}"
                                 for c in unknown_connections]))
    latex_file.write("\n\\end{align*}\n\\[\n")
    latex_file.write(printutils.latex_format_matrix(np.transpose(unknown_connection_dirs), force_vertical=True) + "\n")
    latex_file.write(printutils.latex_format_matrix(dir_names, force_vertical=True) + "\n")
    latex_file.write(" = -")
    latex_file.write(printutils.latex_format_matrix([[resultant_values_x], [resultant_values_y]], force_vertical=True) + "\n")
    latex_file.write(" = -")
    latex_file.write(printutils.latex_format_matrix([[resultant_x], [resultant_y]], force_vertical=True))
    latex_file.write("\\]\n")

    # solve the force matrix
    forces = solve_force_matrix(unknown_connection_dirs, list(known_force_resultant))
    print("Solved forces:")
    latex_file.write("\\begin{align*}\n")
    latex_file.write(printutils.oxford_list(list(zip(unknown_names, forces)), fmt="T_{{{0}}} &= {1:.3f} ", sep=" & ",
                                            comma_on_2=True, final_conjunction=None))
    latex_file.write("\n\\end{align*}\n")
    # store the new data back into the truss model
    for load, connection in zip(forces, unknown_connections):
        connection.set_load(load)
        print(f"{connection.start.name}{connection.end.name}: {connection.load_vec} (coef: {load})")
