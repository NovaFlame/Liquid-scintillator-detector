"""Сферический детектор, все длины заданы в метрах."""

import numpy as np


class Detector:
    def __init__(self, R=2.0, N=40, r=0.10, a=0.5, seed=None):
        if isinstance(N, (bool, np.bool_)) or not isinstance(N, (int, np.integer)):
            raise ValueError("N должно быть целым положительным числом.")
        self.R, self.N, self.r, self.a = float(R), int(N), float(r), float(a)
        if not np.all(np.isfinite([self.R, self.r, self.a])):
            raise ValueError("R, r и a должны быть конечными числами.")
        if self.R <= 0 or self.N <= 0 or not 0 < self.r <= self.R:
            raise ValueError("Требуется R > 0, N > 0 и 0 < r <= R.")
        if self.a < 0:
            raise ValueError("Среднее число фотоэлектронов a не может быть отрицательным.")
        self.rng = np.random.default_rng(seed)

        # Решётка Фибоначчи.
        i = np.arange(self.N)
        z = 1.0 - 2.0 * (i + 0.5) / self.N
        phi = np.pi * (3.0 - np.sqrt(5.0)) * i
        rho = np.sqrt(1.0 - z**2)
        self.pmt_positions = self.R * np.column_stack(
            (rho * np.cos(phi), rho * np.sin(phi), z)
        )

    def find_pmt(self, point, atol=1e-9):
        """Вернуть индекс ФЭУ (от 0 до N-1) или None при отсутствии попадания.

        Точка должна иметь три конечные координаты в метрах.
        Точки вне поверхности сферы возвращают None. Граница ФЭУ включена.
        atol — абсолютный допуск в метрах для обеих проверок.
        Если чувствительные области пересекаются, выбирается ближайший центр.
        """
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("point должна содержать три конечные координаты.")
        if not np.isfinite(atol) or atol < 0:
            raise ValueError("atol должен быть конечным неотрицательным числом.")
        if not np.isclose(np.linalg.norm(point), self.R, atol=atol, rtol=0):
            return None

        distances = np.linalg.norm(self.pmt_positions - point, axis=1)
        index = int(np.argmin(distances))
        return index if distances[index] <= self.r + atol else None

    def is_on_pmt(self, point, atol=1e-9):
        """Принадлежит ли точка поверхности хотя бы одному ФЭУ?"""
        return self.find_pmt(point, atol=atol) is not None

    def boundary_intersection(self, position, direction, atol=1e-9):
        """Первая точка достижения границы фотоном изнутри детектора.

        position и direction — тройки конечных координат. Ненулевое
        направление нормируется автоматически. Источник вне сферы недопустим.
        Если источник на границе (с допуском atol), поглощение немедленное.
        """
        position = np.asarray(position, dtype=float)
        direction = np.asarray(direction, dtype=float)
        for vector in (position, direction):
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                raise ValueError("Положение и направление должны быть конечными тройками.")
        if not np.isfinite(atol) or atol < 0:
            raise ValueError("atol должен быть конечным неотрицательным числом.")

        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0 or not np.isfinite(direction_norm):
            raise ValueError("Направление должно иметь конечную ненулевую длину.")
        direction = direction / direction_norm
        radius = np.linalg.norm(position)
        if radius > self.R + atol:
            raise ValueError("Источник должен находиться внутри детектора или на границе.")
        if radius > 0 and abs(radius - self.R) <= atol:
            return position * (self.R / radius)

        # |position + t * direction|^2 = R^2, где t >= 0.
        b = np.dot(position, direction)
        c = np.dot(position, position) - self.R**2
        root = np.sqrt(max(0.0, b**2 - c))
        # Эквивалент t = -b + root, с устойчивой формой при b >= 0.
        t = -c / (b + root) if b >= 0 else -b + root
        return position + t * direction

    def trace_photon(self, position, direction, atol=1e-9):
        """Вернуть индекс ФЭУ или None после первого попадания на границу.

        При промахе фотон поглощён: дальнейших отражений не происходит.
        """
        point = self.boundary_intersection(position, direction, atol=atol)
        return self.find_pmt(point, atol=atol)

    def sample_photoelectrons(self, size=None):
        """Число фотоэлектронов на один попавший фотон: Poisson(a).

        Без size возвращается одно целое число; с size — массив выборки.
        Вызывать только для фотонов, достигших фотокатода.
        """
        return self.rng.poisson(self.a, size=size)

    def visualize(self):
        """Вернуть k3d.Plot: каркас сферы и чувствительные области ФЭУ.

        В Jupyter вызовите detector.visualize().display().
        k3d импортируется только при вызове этого метода.
        """
        try:
            import k3d
        except ImportError as exc:
            raise ImportError("Для визуализации установите k3d: pip install k3d") from exc

        n_lat, n_lon = 32, 64
        theta = np.linspace(0.0, np.pi, n_lat + 1)
        phi = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
        theta, phi = np.meshgrid(theta, phi, indexing="ij")
        vertices = self.R * np.column_stack(
            (
                (np.sin(theta) * np.cos(phi)).ravel(),
                (np.sin(theta) * np.sin(phi)).ravel(),
                np.cos(theta).ravel(),
            )
        )
        faces = []
        for i in range(n_lat):
            for j in range(n_lon):
                left = i * n_lon + j
                right = i * n_lon + (j + 1) % n_lon
                below_left, below_right = left + n_lon, right + n_lon
                if i != 0:
                    faces.append((left, below_left, right))
                if i != n_lat - 1:
                    faces.append((right, below_left, below_right))

        plot = k3d.plot()
        plot += k3d.mesh(
            vertices.astype(np.float32), np.asarray(faces, dtype=np.uint32),
            color=0x7B9CB5, wireframe=True, opacity=0.20, name="Detector",
        )

        # Для хорды r угловой радиус равен 2 arcsin(r / (2R)).
        alpha = 2.0 * np.arcsin(self.r / (2.0 * self.R))
        n_rim = 48
        angles = np.linspace(0.0, 2.0 * np.pi, n_rim, endpoint=False)
        cap_vertices, cap_faces = [], []
        for position in self.pmt_positions:
            normal = position / self.R
            reference = np.array([0.0, 0.0, 1.0])
            if abs(normal[2]) > 0.9:
                reference = np.array([1.0, 0.0, 0.0])
            u = np.cross(normal, reference)
            u /= np.linalg.norm(u)
            v = np.cross(normal, u)
            rim = (
                np.cos(alpha) * normal
                + np.sin(alpha)
                * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)
            )
            offset = len(cap_vertices)
            # Малое смещение только для изображения предотвращает мерцание сеток.
            cap_vertices.extend(1.001 * self.R * np.vstack((normal, rim)))
            for j in range(n_rim):
                cap_faces.append((offset, offset + 1 + j, offset + 1 + (j + 1) % n_rim))

        plot += k3d.mesh(
            np.asarray(cap_vertices, dtype=np.float32),
            np.asarray(cap_faces, dtype=np.uint32),
            color=0xE49B36, side="double", name="Photomultipliers",
        )
        return plot


if __name__ == "__main__":
    detector = Detector()
    print("Координаты ФЭУ, м:")
    print(detector.pmt_positions)
    point = detector.pmt_positions[0]
    print("Индекс ФЭУ для его центра:", detector.find_pmt(point))
    print("Точка (0, 0, 0) принадлежит ФЭУ:", detector.is_on_pmt([0, 0, 0]))
