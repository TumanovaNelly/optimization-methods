import tkinter as tk
from dataclasses import dataclass, field
from tkinter import ttk, messagebox
from typing import List

import numpy as np


class SmartEntry(ttk.Entry):
    PLACEHOLDER = "0"

    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.insert(0, self.PLACEHOLDER)
        self._has_placeholder = True
        self.on_arrow = None

        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<FocusOut>", self._on_focus_out)
        self.bind("<Left>", lambda e: self._arrow("left"))
        self.bind("<Right>", lambda e: self._arrow("right"))
        self.bind("<Up>", lambda e: self._arrow("up"))
        self.bind("<Down>", lambda e: self._arrow("down"))

    def _on_focus_in(self, event=None):
        if self._has_placeholder and self.get().strip() == self.PLACEHOLDER:
            self.delete(0, tk.END)
            self._has_placeholder = False
        else:
            self.select_range(0, tk.END)

    def _on_focus_out(self, event=None):
        if self.get().strip() == "":
            self.delete(0, tk.END)
            self.insert(0, self.PLACEHOLDER)
            self._has_placeholder = True

    def get_value(self):
        val = self.get().strip()
        return "0" if val == "" else val

    def set_value(self, value):
        self.delete(0, tk.END)
        if value is None or value == "":
            self.insert(0, self.PLACEHOLDER)
            self._has_placeholder = True
        else:
            self.insert(0, value)
            self._has_placeholder = (str(value).strip() == self.PLACEHOLDER)

    def _arrow(self, direction):
        if self.on_arrow is not None and self.on_arrow(self, direction):
            return "break"
        return None


@dataclass
class SimplexData:
    """Сырые данные, собранные из GUI (все значения — строки)."""
    optimization_type: str = "max"                                     # "max" / "min"
    obj_func_vector: List[str] = field(default_factory=list)           # коэффициенты ЦФ
    constraints_matrix: List[List[str]] = field(default_factory=list)  # матрица ограничений
    signs: List[str] = field(default_factory=list)                     # знаки
    constraints_const_vector: List[str] = field(default_factory=list)  # правые части
    num_vars: int = 0
    num_constraints: int = 0


class SimplexProcessor:
    """
    Отвечает за:
    - проверку данных и преобразование строк в float;
    - вывод задачи в консоль;
    - (в будущем) реализацию симплекс-метода.
    """

    @staticmethod
    def _to_float(s: str) -> float:
        """Преобразует строку в float, поддерживая запятую как разделитель."""
        return float(s.replace(",", "."))

    def print_task(self, data: SimplexData) -> None:
        """Печатает задачу в консоль."""
        print("Целевая функция:")
        terms = " + ".join(f"{coef} * x{j + 1}" for j, coef in enumerate(data.obj_func_vector))
        print(f"    F = {terms}  →  {data.optimization_type}")
        print("Ограничения:")
        for i in range(data.num_constraints):
            terms = " + ".join(f"{a} * x{j + 1}"
                               for j, a in enumerate(data.constraints_matrix[i]))
            print(f"    {terms} {data.signs[i]} {data.constraints_const_vector[i]}")
        print("Условия неотрицательности: xⱼ ≥ 0")
        print("\n\n")

    def solve(self, data: SimplexData):
        """
        Здесь позже можно реализовать сам симплекс-метод.
        Сейчас — заглушка.
        """
        # Целевая
        z = np.array(data.obj_func_vector).astype(np.float64)
        if data.optimization_type == "min":
            z = -z

        # Индексы базисных векторов
        base_indexes = []

        # Приводим матрицу ограничений к канону
        f = np.array(data.constraints_matrix).astype(np.float64)
        for i in range(data.num_constraints):
            if data.signs[i] == ">=":
                new_col = np.zeros(data.num_constraints)
                new_col[i] = -1
                f = np.column_stack((f, new_col))

            basis_column = np.zeros(data.num_constraints)
            basis_column[i] = 1
            base_indexes.append(f.shape[1])
            f = np.column_stack((f, basis_column))

        padding = np.zeros(f.shape[1] - len(z))
        z = np.append(z, padding)

        f_expanded = np.column_stack((f, data.constraints_const_vector)).astype(np.float64)

        # текущий максимум z
        z_mx = -np.inf

        while True:
            # пересчитываем нижнюю строку
            d = np.array([-(f_expanded[:, i] @ z[base_indexes]) + z[i] for i in range(len(z))])
            z_mx = f_expanded[:, -1] @ z[base_indexes]

            # находим разрешающий элемент
            col_index = np.argmax(d)
            if d[col_index] <= 0.0: break
            row_index = 0
            cur_min = np.inf
            for i in range(len(f_expanded)):
                if f_expanded[i][col_index] <= 0.0: continue
                val = f_expanded[i][-1] / f_expanded[i][col_index]
                if cur_min > val:
                    cur_min = val
                    row_index = i

            # меняем базис
            base_indexes[row_index] = col_index
            f_expanded[row_index] /= f_expanded[row_index][col_index]
            for i in range(len(f_expanded)):
                if i == row_index: continue
                f_expanded[i] -= f_expanded[row_index] * f_expanded[i][col_index]

        answer = np.zeros_like(z)
        answer[base_indexes] = f_expanded[:, -1]

        print("РЕЗУЛЬТАТ:")
        for i in range(data.num_vars):
            print(f"    x{i + 1} = {answer[i]}")

        print(f"F = {z_mx}")
        print("\n\n")


class SimplexInputApp:
    PAD_ENTRY = (2, 2)
    PAD_LABEL = (2, 2)
    PAD_SIGN = (10, 10)

    def __init__(self, root, processor: SimplexProcessor = None):
        self.root = root
        self.root.title("Ввод данных для симплекс-метода")
        self.root.geometry("520x520")

        self.processor = processor or SimplexProcessor()

        self.objective_entries = []
        self.constraint_entries = []
        self.constraint_signs = []
        self.rhs_entries = []

        self.grid = []
        self.row_rhs = []

        self.saved_objective = []
        self.saved_constraints = []
        self.saved_signs = []
        self.saved_rhs = []

        self._skip_save = False

        self.num_vars = tk.IntVar(value=2)
        self.num_constraints = tk.IntVar(value=2)
        self.optim_type = tk.StringVar(value="max")

        self.num_vars.trace_add("write", self.on_params_changed)
        self.num_constraints.trace_add("write", self.on_params_changed)
        self.optim_type.trace_add("write", self.on_params_changed)

        self.build_ui()
        self.rebuild_input_area()

    # ---------- Интерфейс ----------
    def build_ui(self):
        # Параметры
        self.params_frame = ttk.LabelFrame(self.root, text="Параметры задачи",
                                           padding=10)
        self.params_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(self.params_frame,
                  text="Число переменных:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(self.params_frame, from_=1, to=15,
                    textvariable=self.num_vars,
                    width=5).grid(row=0, column=1, padx=5, sticky="w")

        ttk.Label(self.params_frame,
                  text="Число ограничений:").grid(row=0, column=2,
                                                  sticky="w", padx=(20, 0))
        ttk.Spinbox(self.params_frame, from_=1, to=15,
                    textvariable=self.num_constraints,
                    width=5).grid(row=0, column=3, padx=5, sticky="w")

        ttk.Label(self.params_frame,
                  text="Тип оптимизации:").grid(row=1, column=0,
                                                sticky="w", pady=(8, 0))
        ttk.Radiobutton(self.params_frame, text="Максимизация",
                        variable=self.optim_type, value="max",
                        command=self.rebuild_input_area).grid(
            row=1, column=1, sticky="w", pady=(8, 0))
        ttk.Radiobutton(self.params_frame, text="Минимизация",
                        variable=self.optim_type, value="min",
                        command=self.rebuild_input_area).grid(
            row=1, column=2, sticky="w", pady=(8, 0))

        # Целевая функция
        self.obj_frame = ttk.LabelFrame(self.root, text="Целевая функция",
                                        padding=10)
        self.obj_frame.pack(fill="x", padx=10, pady=5)

        # Ограничения
        self.con_frame = ttk.LabelFrame(self.root, text="Ограничения",
                                        padding=10)
        self.con_frame.pack(fill="x", padx=10, pady=5)

        # Кнопки
        bottom = ttk.Frame(self.root)
        bottom.pack(fill="x", padx=10, pady=5)

        ttk.Button(bottom, text="Считать данные",
                   command=self.collect_data).pack(side="left")
        ttk.Button(bottom, text="Очистить",
                   command=self.clear_all).pack(side="left", padx=5)

    # ---------- Реакция на изменения ----------
    def on_params_changed(self, *args):
        self.root.after(50, self.rebuild_input_area)

    # ---------- Сохранение значений ----------
    def save_current_values(self):
        if self._skip_save:
            return
        self.saved_objective = [e.get_value() for e in self.objective_entries]
        self.saved_constraints = [[e.get_value() for e in row]
                                  for row in self.constraint_entries]
        self.saved_signs = [s.get() for s in self.constraint_signs]
        self.saved_rhs = [e.get_value() for e in self.rhs_entries]

    # ---------- Навигация ----------
    def _make_navigator(self):
        def navigate(entry, direction):
            pos = None
            for r, row in enumerate(self.grid):
                for c, e in enumerate(row):
                    if e is entry:
                        pos = (r, c)
                        break
                if pos:
                    break
            if pos is None:
                return False

            r, c = pos
            max_row = len(self.grid) - 1

            if direction == "left":
                if c > 0:
                    self.grid[r][c - 1].focus_set()
                    return True
                if r > 0:
                    self.grid[r - 1][len(self.grid[r - 1]) - 1].focus_set()
                    return True
                return False

            if direction == "right":
                if c < len(self.grid[r]) - 1:
                    self.grid[r][c + 1].focus_set()
                    return True
                if r < max_row:
                    self.grid[r + 1][0].focus_set()
                    return True
                return False

            if direction == "up":
                if r > 0:
                    target_col = min(c, len(self.grid[r - 1]) - 1)
                    self.grid[r - 1][target_col].focus_set()
                    return True
                return False

            if direction == "down":
                if r < max_row:
                    target_col = min(c, len(self.grid[r + 1]) - 1)
                    self.grid[r + 1][target_col].focus_set()
                    return True
                return False

            return False

        return navigate

    # ---------- Настройка колонок ----------
    def _setup_frame_columns(self, frame, total_columns):
        for c in range(total_columns):
            frame.columnconfigure(c, weight=0)
        frame.columnconfigure(total_columns, weight=1)

    # ---------- Строка формулы ----------
    def _build_formula_row(self, parent, row_index, entries_row,
                           saved_row, sign_widget=None, rhs_entry=None):
        grid_row = []
        col = 0
        n = len(entries_row)

        for j in range(n):
            e = entries_row[j]
            if saved_row is not None and j < len(saved_row):
                e.set_value(saved_row[j])
            e.grid(row=row_index, column=col,
                   padx=self.PAD_ENTRY, pady=3, sticky="w")
            grid_row.append(e)
            col += 1

            ttk.Label(parent, text=f"x{j + 1}").grid(
                row=row_index, column=col,
                padx=self.PAD_LABEL, pady=3, sticky="w")
            col += 1

            if j < n - 1:
                ttk.Label(parent, text="+").grid(
                    row=row_index, column=col,
                    padx=self.PAD_LABEL, pady=3, sticky="w")
                col += 1

        if sign_widget is not None:
            sign_widget.grid(row=row_index, column=col,
                             padx=self.PAD_SIGN, pady=3, sticky="w")
            col += 1

        if rhs_entry is not None:
            rhs_entry.grid(row=row_index, column=col,
                           padx=self.PAD_ENTRY, pady=3, sticky="w")
            grid_row.append(rhs_entry)
            col += 1

        return grid_row, col

    # ---------- Перестроение ----------
    def rebuild_input_area(self):
        self.save_current_values()

        for w in self.obj_frame.winfo_children():
            w.destroy()
        for w in self.con_frame.winfo_children():
            w.destroy()

        self.objective_entries.clear()
        self.constraint_entries.clear()
        self.constraint_signs.clear()
        self.rhs_entries.clear()
        self.grid.clear()
        self.row_rhs.clear()

        n = self.num_vars.get()
        m = self.num_constraints.get()

        opt_label = "max" if self.optim_type.get() == "max" else "min"
        self.obj_frame.config(text=f"Целевая функция  (F → {opt_label})")

        navigator = self._make_navigator()

        def make_entry(parent):
            e = SmartEntry(parent, width=8, justify="center")
            e.on_arrow = navigator
            return e

        # ЦФ
        obj_entries = [make_entry(self.obj_frame) for _ in range(n)]
        obj_grid_row, obj_cols = self._build_formula_row(
            self.obj_frame, 0, obj_entries, self.saved_objective)
        self._setup_frame_columns(self.obj_frame, obj_cols)

        self.objective_entries.extend(obj_entries)
        self.grid.append(obj_grid_row)

        # Ограничения
        max_cols = 0
        for i in range(m):
            row_entries = [make_entry(self.con_frame) for _ in range(n)]

            sign = ttk.Combobox(self.con_frame, values=["<=", ">=", "="],
                                width=3, state="readonly")
            if i < len(self.saved_signs) and self.saved_signs[i] in ("<=", ">=", "="):
                sign.set(self.saved_signs[i])
            else:
                sign.set("<=")

            rhs = make_entry(self.con_frame)

            grid_row, cols = self._build_formula_row(
                self.con_frame, i, row_entries,
                self.saved_constraints[i] if i < len(self.saved_constraints) else None,
                sign_widget=sign, rhs_entry=rhs)

            if i < len(self.saved_rhs):
                rhs.set_value(self.saved_rhs[i])

            self.constraint_entries.append(row_entries)
            self.constraint_signs.append(sign)
            self.rhs_entries.append(rhs)
            self.row_rhs.append(rhs)
            self.grid.append(grid_row)

            max_cols = max(max_cols, cols)

        self._setup_frame_columns(self.con_frame, max_cols)

        ttk.Label(self.con_frame,
                  text="Условия неотрицательности: x₁ ≥ 0, x₂ ≥ 0, …  ",
                  foreground="gray").grid(
            row=m, column=0, columnspan=max_cols + 1,
            sticky="w", pady=(10, 0))

    # ---------- Очистка ----------
    def clear_all(self):
        for e in self.objective_entries:
            e.set_value("")
        for row in self.constraint_entries:
            for e in row:
                e.set_value("")
        for e in self.rhs_entries:
            e.set_value("")
        for s in self.constraint_signs:
            s.set("<=")

        self.saved_objective = []
        self.saved_constraints = []
        self.saved_signs = []
        self.saved_rhs = []

        self._skip_save = True
        try:
            self.rebuild_input_area()
        finally:
            self._skip_save = False

    # ---------- Считывание (только сбор данных, без обработки) ----------
    def collect_data(self):
        """Собирает сырые строки из полей и отдаёт их процессору."""
        raw = SimplexData(
            optimization_type=self.optim_type.get(),
            obj_func_vector=[e.get_value() for e in self.objective_entries],
            constraints_matrix=[[e.get_value() for e in row] for row in self.constraint_entries],
            signs=[s.get() for s in self.constraint_signs],
            constraints_const_vector=[e.get_value() for e in self.rhs_entries],
            num_vars=self.num_vars.get(),
            num_constraints=self.num_constraints.get(),
        )

        self.processor.print_task(raw)
        self.processor.solve(raw)

        messagebox.showinfo("Данные считаны",
                            "Результат выведен в консоль")


if __name__ == "__main__":
    root = tk.Tk()
    app = SimplexInputApp(root)
    root.mainloop()
