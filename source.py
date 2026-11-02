"""Изотропный источник фотонов с линейным световым выходом."""

import numpy as np


class Source:
    def __init__(self, Y=10000, seed=None):
        self.Y = float(Y)
        if not np.isfinite(self.Y) or self.Y <= 0:
            raise ValueError("Y должен быть конечным положительным числом.")
        self.rng = np.random.default_rng(seed)

    def sample_directions(self, E):
        """Вернуть массив единичных направлений формы (round(Y * E), 3).

        Каждая строка — независимое направление (nx, ny, nz).
        При E = 0 возвращается пустой массив формы (0, 3).
        """
        E = float(E)
        if not np.isfinite(E) or E < 0:
            raise ValueError("E должна быть конечной неотрицательной энергией в МэВ.")

        N = int(round(self.Y * E))
        # Для изотропии равномерным должен быть cos(theta), а не theta.
        cos_theta = self.rng.uniform(-1.0, 1.0, size=N)
        phi = self.rng.uniform(0.0, 2.0 * np.pi, size=N)
        sin_theta = np.sqrt(1.0 - cos_theta**2)

        return np.column_stack(
            (sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta)
        )


if __name__ == "__main__":
    source = Source(seed=42)
    directions = source.sample_directions(E=2.0)
    print("Число фотонов:", len(directions))
    print("Первые пять направлений:")
    print(directions[:5])
