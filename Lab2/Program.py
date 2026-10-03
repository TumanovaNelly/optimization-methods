import heapq
import time
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application

from dataclasses import dataclass
from collections.abc import Callable
from typing import List, Tuple


@dataclass
class Point:
    x: float
    y: float

    def __str__(self):
        return f"({self.x}, {self.y})"


@dataclass
class FindMinResult:
    min_point: Point
    error: Point
    broken_line_points: List[Point]
    time: float
    iter_count: int

    def plot_result(self, func: Callable[[float], float], func_label, borders: Tuple[float, float]):
        left, right = sorted(borders)
        func_xs = np.linspace(left, right, 1000)
        func_ys = list(map(func, func_xs))
        plt.plot(func_xs, func_ys, label=func_label, color="tab:blue", lw=2, zorder=3)

        plt.plot([point.x for point in self.broken_line_points],
                 [point.y for point in self.broken_line_points],
                 label="Broken line", color="tab:orange", ls=":", lw=2, zorder=2)

        plt.scatter(self.min_point.x,
                    self.min_point.y,
                    label=f"Minimum point ({self.min_point.x:.2f}, {self.min_point.y:.2f})", color="tab:red", zorder=4)

        plt.legend()
        plt.grid(True)
        plt.show()

    def __str__(self):
        return \
f"""The minimum value y = {self.min_point.y} is achieved at the point x = {self.min_point.x}
△x = {self.error.x}
△y = {self.error.y}
Number of iterations: {self.iter_count}
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
        self.func = func

    def solve(self, borders: Tuple[float, float], eps: float = 1e-4, L: float = 100) -> FindMinResult:
        left, right = sorted(borders)
        left_point = Point(left, self.func(left))
        right_point = Point(right, self.func(right))

        start_time = time.perf_counter()

        first_corner = self.find_corner((left_point, right_point), L)
        corners = [first_corner]
        heapq.heapify(corners)

        cur_min = Point(0, float('inf'))
        iterations = 0
        while True:
            min_corner = heapq.heappop(corners)
            if (cur_min.y - min_corner.center.y) < eps: break
            iterations += 1
            left_sub_corner = self.find_corner((min_corner.left, min_corner.center_func), L)
            right_sub_corner = self.find_corner((min_corner.center_func, min_corner.right), L)
            cur_min = min(cur_min, left_sub_corner.center_func, right_sub_corner.center_func,
                          key=lambda p: p.y)
            heapq.heappush(corners, left_sub_corner)
            heapq.heappush(corners, right_sub_corner)

        end_time = time.perf_counter()

        broken_line_points = [Point(left, self.func(left))]
        for corner in corners:
            broken_line_points.append(corner.center)
            broken_line_points.append(corner.right)
        broken_line_points.sort(key=lambda p: p.x)

        return FindMinResult(
            min_point=cur_min,
            error=Point(eps / L, eps),
            broken_line_points=broken_line_points,
            time=end_time - start_time,
            iter_count=iterations)

    def find_corner(self, points: Tuple[Point, Point], slope: float) -> Corner:
        left_point, right_point = sorted(points, key=lambda p: p.x)
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
    expr = sp.parse_expr(expr, local_dict={'x': x}, transformations=transforms, evaluate=True)

    return sp.lambdify(x, expr, 'numpy')


if __name__ == "__main__":
    while True:
        print("Enter expression or 'exit':")
        expression = input().strip().replace(" ", "")
        if expression == "exit": break
        try:
            function = make_func_from_string(expression)
        except Exception:
            print("Error while parsing expression")
            print()
            continue

        while True:
            print("Enter borders separated by spaces or 'exit':")
            borders = input().strip()
            if borders == "exit": break
            try:
                borders = tuple(map(float, borders.split(" ")))
            except Exception:
                print("Error while parsing borders")
                continue

            if len(borders) != 2:
                print("Invalid number of borders")
                continue

            problem = FindMinProblem(function)
            result = problem.solve(borders=(borders[0], borders[1]))
            print(result)
            print()
            result.plot_result(function, f"y={expression}", (borders[0], borders[1]))
