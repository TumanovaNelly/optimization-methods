from dataclasses import dataclass, field
from typing import Optional, List, Dict

import sympy as sp
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application

from color_print import *


@dataclass
class Function:
    name: str
    expression: sp.Expr

    def __str__(self):
        return f"{self.name} = {self.expression}"


@dataclass
class Point:
    coordinates: Dict[sp.Symbol, sp.Expr]
    func_value: sp.Expr
    func_str: str
    limitation_values: List[sp.Expr]
    limitation_strs: List[str]
    labels: List[str] = field(default_factory=list)

    def __str__(self):
        text = []
        for label in self.labels:
            text.append(label)
        text.append("")

        for sym, expr in self.coordinates.items():
            text.append(f"{sym} = {expr}")
        text.append("")

        text.append(f"{self.func_str} = {self.func_value}")

        for string, value in zip(self.limitation_strs, self.limitation_values):
            text.append(f"{string} = {value}")
        text.append("")

        return "\n".join(text)


@dataclass
class FindExtrOkResult:
    L: sp.Expr
    grad_L: List[sp.Expr]
    candidates: List[Point]

    def __str__(self):
        text = []
        text.append(success("THE RESULT OF THE LAGRANGE MULTIPLIER METHOD"))
        text.append(success("————————————————————————————————————————————"))
        for i, point in enumerate(self.candidates):
            text.append(f"-------------- CANDIDATE {i} --------------")
            text.append(str(point))
        text.append(success("————————————————————————————————————————————"))
        text.append("")
        return "\n".join(text)


class FindExtrProblem:
    def __init__(self, func: Function, lims: List[Function], variables: List[sp.Symbol]):
        self.function = func
        self.limitations = lims
        self.variables = variables

    def solve(self) -> FindExtrOkResult:
        lambdas = sp.symbols(f"lambda:{len(self.limitations)}", seq=True)
        all_variables = self.variables + lambdas

        L = self.function.expression
        for lamb, limitation in zip(lambdas, self.limitations):
            L += lamb * limitation.expression

        grad_L = ([sp.diff(L, var) for var in self.variables] +
                  [limitation.expression for limitation in self.limitations])

        solutions = sp.solve(grad_L, all_variables, dict=True, seq=True)
        candidates = self.__make_candidates(solutions)

        return FindExtrOkResult(
            L=L, grad_L=grad_L,
            candidates=candidates,
        )

    def __str__(self):
        limits_strs = '\n'.join(map(str, self.limitations))
        return \
f"""
Find the extremes of the function {self.function} under constraints:
{limits_strs}
"""

    def __make_candidates(self, solutions: List[Dict[sp.Symbol, sp.Expr]]) -> List[Point]:
        func_str = str(self.function)
        lim_strs = list(map(str, self.limitations))
        candidates = [Point(
            coordinates=cand,
            func_value=self.function.expression.subs(cand),
            func_str=func_str,
            limitation_values=[limitation.expression.subs(cand) for limitation in self.limitations],
            limitation_strs=lim_strs,
        ) for cand in solutions]

        min_func_val = min(p.func_value for p in candidates)
        max_func_val = max(p.func_value for p in candidates)

        jacobian = sp.Matrix([[sp.diff(lim.expression, var) for var in self.variables] for lim in self.limitations])

        for cand in candidates:
            if cand.func_value == min_func_val: cand.labels.append(label("[GLOBAL MINIMUM]"))
            if cand.func_value == max_func_val: cand.labels.append(label("[GLOBAL MAXIMUM]"))

            J_val = jacobian.subs(cand.coordinates)
            rank_val = J_val.rank()
            max_rank = min(len(self.limitations), len(self.variables))
            if rank_val < max_rank:
                cand.labels.append(
                    warning(f"[THE RANK OF THE JACOBI MATRIX ({rank_val}) IS LESS THAN EXPECTED ({max_rank})]"))

        return candidates


def ask_ok() -> bool:
    while True:
        print("Is everything correct? Enter y or n:")
        y_n = input().strip().lower()
        if y_n == "n": return False
        if y_n == "y": return True


def read_number(of="") -> Optional[int]:
    number = None
    if of: of = "of " + of
    while True:
        print(f"Enter number {of} or 'exit':")
        try:
            number = int(input())
        except ValueError:
            print_error("Invalid number")
            continue
        break
    return number


def read_variables() -> Optional[List[sp.Symbol]]:
    variables = None
    while True:
        print("Enter variables separated by spaces or 'exit':")
        print_label(f"[FORMAT] x1:3 = x1 x2 x3")
        print_label(f"[FORMAT] x:2 = x0 x1 x2")

        vars_str = input()
        if vars_str == "exit": break

        try:
            variables = sp.symbols(vars_str, seq=True)
        except ValueError as ex:
            print_error(ex)
            continue

        for i, variable in enumerate(variables):
            print_warning(f"Variable {i}: {variable}")

        if ask_ok(): break

    return variables


def read_expression(name: str, variables: List[sp.Symbol]) -> Optional[Function]:
    function = None
    while True:
        print(f"Enter {name} or 'exit':")
        expr_str = input()
        if expr_str == "exit": break

        try:
            expr = expression_from_string(expr_str, variables)
        except ValueError as ex:
            print_error(ex)
            continue

        function = Function(name=name, expression=expr)
        break

    return function


def expression_from_string(string: str, variables: List[sp.Symbol]) -> sp.Expr:
    transforms = standard_transformations + (implicit_multiplication_application,)
    expression = sp.parse_expr(string, local_dict={v.name: v for v in variables}, transformations=transforms,
                               evaluate=True)

    free = expression.free_symbols.difference(variables)
    if free: raise ValueError(f"Unknown symbols in expression: {free}")

    return expression


if __name__ == "__main__":
    while True:
        variables = read_variables()
        if variables is None: break
        vars_str = ", ".join(map(str, variables))

        f = read_expression(f"f({vars_str})", variables)
        if f is None: break

        h_num = read_number("limitations")
        if h_num is None: break

        hs = []
        was_exit = False
        for i in range(h_num):
            h_i = read_expression(f"h_{i}({vars_str})", variables)
            if h_i is None:
                was_exit = True
                break
            hs.append(h_i)
        if was_exit: break

        problem = FindExtrProblem(f, hs, variables)
        print(problem)

        try:
            result = problem.solve()
        except ValueError as ex:
            print_error(ex)
            continue

        print(result)
