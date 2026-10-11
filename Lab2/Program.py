import heapq
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application


RED = "\033[91m"
BASE_COLOR = "\033[0m"

@dataclass
class Point:
    x: float
    y: float

    def __str__(self):
        return f"({self.x}, {self.y})"


@dataclass
class FindMinResult:
    min_point: Point
    broken_line_points: List[Point]
    potential_mins: List[Point]
    time: float
    iter_count: int
    max_iter: int
    L: float
    eps: float

    def plot_result(self, func: Callable[[float], float], func_label, borders: Tuple[float, float]):
        left, right = sorted(borders)
        func_xs = np.linspace(left, right, 1000)
        func_ys = list(map(func, func_xs))
        plt.plot(func_xs, func_ys, label=func_label, color="tab:blue", lw=2, zorder=3)

        plt.plot([point.x for point in self.broken_line_points],
                 [point.y for point in self.broken_line_points],
                 label=f"Broken line (L={self.L:.2f})", color="tab:orange", ls=":", lw=2, zorder=2)

        plt.scatter([point.x for point in self.potential_mins],
                    [point.y for point in self.potential_mins],
                    label=f"Potential minimum points", color="tab:red", s=5, zorder=4)

        plt.scatter(self.min_point.x,
                    self.min_point.y,
                    label=f"Minimum point ({self.min_point.x:.2f}, {self.min_point.y:.2f})", color="tab:red", zorder=5)

        plt.legend()
        plt.grid(True)
        plt.show()

    def __str__(self):
        return \
f"""The minimum value y = {self.min_point.y} is achieved at the point x = {self.min_point.x}
L = {self.L}
ɛ = {self.eps}
Number of iterations: {self.iter_count} out of {self.max_iter}
Execution time: {self.time}"""


class FindMinProblem:
    @dataclass
    class Corner:
        left: Point
        right: Point
        center: Point
        center_func: Point

        def __lt__(self, other):
            return self.center.y < other.center.y

    def __init__(self, func: Callable[[float], float]):
        self.func = self.__add_finite_error_raise(func)

    @staticmethod
    def __add_finite_error_raise(func):
        def safe(x):
            y = func(x)
            if not np.isfinite(y):
                raise ValueError(f"Function value is not finite (x = {x}, y = {y})")
            return y

        return safe

    def solve(self, borders: Tuple[float, float], eps: float = 1e-4, max_iter: int = 10000) -> FindMinResult:
        left, right = sorted(borders)
        L = self.__find_L(left, right)

        left_point = Point(left, self.func(left))
        right_point = Point(right, self.func(right))

        start_time = time.perf_counter()

        first_corner = self.__find_corner(left_point, right_point, L)
        corners = [first_corner]
        heapq.heapify(corners)

        cur_min = min(left_point, first_corner.center_func, right_point,
                          key=lambda p: p.y)
        iterations = 0
        for _ in range(max_iter):
            min_corner = corners[0]
            if (cur_min.y - min_corner.center.y) < eps: break
            iterations += 1
            heapq.heappop(corners)
            left_sub_corner = self.__find_corner(min_corner.left, min_corner.center_func, L)
            right_sub_corner = self.__find_corner(min_corner.center_func, min_corner.right, L)
            cur_min = min(cur_min, left_sub_corner.center_func, right_sub_corner.center_func,
                          key=lambda p: p.y)
            heapq.heappush(corners, left_sub_corner)
            heapq.heappush(corners, right_sub_corner)

        end_time = time.perf_counter()

        broken_line_points = [left_point]
        potential_mins = []
        for corner in corners:
            broken_line_points.append(corner.center)
            broken_line_points.append(corner.right)
            potential_mins.append(corner.center_func)
        broken_line_points.sort(key=lambda p: p.x)

        return FindMinResult(
            min_point=cur_min,
            broken_line_points=broken_line_points,
            potential_mins=potential_mins,
            time=end_time - start_time,
            iter_count=iterations, max_iter=max_iter,
            L=L, eps=eps)

    def __find_L(self, left_border: float, right_border: float, intervals_count: int = 1000,
                 reliability: float = 2.0) -> float:
        xs = list(np.linspace(left_border, right_border, intervals_count))
        ys = list(map(self.func, xs))
        mx_slope = 1.0
        for i in range(1, intervals_count):
            mx_slope = max(mx_slope, abs(ys[i] - ys[i - 1]) / (xs[i] - xs[i - 1]))
        return mx_slope * reliability

    def __find_corner(self, left_point: Point, right_point: Point, slope: float) -> Corner:
        x = (left_point.x + right_point.x + (left_point.y - right_point.y) / slope) / 2
        y = (slope * (left_point.x - right_point.x) + left_point.y + right_point.y) / 2
        return self.Corner(
            left=left_point,
            right=right_point,
            center=Point(x, y),
            center_func=Point(x, self.func(x)))


def make_func_from_string(expr: str) -> Callable[[float], float]:
    x = sp.symbols('x')
    transforms = standard_transformations + (implicit_multiplication_application,)
    parsed = sp.parse_expr(expr, local_dict={'x': x}, transformations=transforms, evaluate=True)

    if not parsed.has(x):
        raise ValueError("Expression does not depend on x")

    free = parsed.free_symbols - {x}
    if free:
        raise ValueError(f"Unknown symbols in expression: {free}")

    func = sp.lambdify(x, parsed, 'numpy')

    try: func(1.0)
    except Exception as ex:
        raise ValueError(f"Expression function is not evaluable: {ex}")

    return func


if __name__ == "__main__":
    while True:
        print("Enter expression or 'exit':")
        expression = input().strip().replace(" ", "")
        if expression == "exit": break
        try:
            function = make_func_from_string(expression)
        except ValueError as ex:
            print(f"{RED}{ex}{BASE_COLOR}")
            print()
            continue

        while True:
            print("Enter borders separated by spaces or 'exit':")
            borders = input().strip()
            if borders == "exit": break
            try:
                borders = tuple(map(float, borders.split(" ")))
            except ValueError as ex:
                print(f"{RED}{ex}{BASE_COLOR}")
                print()
                continue

            if len(borders) != 2:
                print(f"{RED}Invalid number of borders (2 was expected){BASE_COLOR}")
                print()
                continue

            problem = FindMinProblem(function)
            try: result = problem.solve(borders=(borders[0], borders[1]))
            except ValueError as ex:
                print(f"{RED}{ex}{BASE_COLOR}")
                print()
                continue

            print(result)
            print()
            result.plot_result(function, f"y={expression}", (borders[0], borders[1]))
