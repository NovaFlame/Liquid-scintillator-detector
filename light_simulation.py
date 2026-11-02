"""Распространение света и заряд сферического детектора."""

import numpy as np

from detector import Detector
from source import Source


def simulate_event(detector, source, E=1.0, position=(0.0, 0.0, 0.0)):
    """Смоделировать событие; вернуть попадания и фотоэлектроны по ФЭУ."""
    directions = source.sample_directions(E)
    hits = np.zeros(detector.N, dtype=np.int64)
    photoelectrons = np.zeros(detector.N, dtype=np.int64)

    for direction in directions:
        index = detector.trace_photon(position, direction)
        # Индекс 0 — допустимый номер ФЭУ, поэтому проверяем именно None.
        if index is not None:
            hits[index] += 1
            photoelectrons[index] += detector.sample_photoelectrons()

    return {
        "photon_count": len(directions),
        "hits": hits,
        "photoelectrons": photoelectrons,
        "Q": int(photoelectrons.sum()),
    }


if __name__ == "__main__":
    source = Source(Y=10000, seed=42)
    detector = Detector(R=2.0, N=40, r=0.10, a=0.5, seed=43)
    event = simulate_event(detector, source, E=1.0, position=(0.0, 0.0, 0.0))

    # Для существующей модели r измеряется по хорде. Площадь сферической
    # чувствительной области в этом случае в точности равна pi * r**2.
    p = detector.N * detector.r**2 / (4.0 * detector.R**2)
    mean_Q = event["photon_count"] * p * detector.a
    var_Q = event["photon_count"] * (
        p * detector.a + p * (1.0 - p) * detector.a**2
    )
    sigma_Q = np.sqrt(var_Q)
    formula_in_statement = (
        detector.a * source.Y * detector.N * detector.r**2 / detector.R**2
    )

    print("Число испущенных фотонов:", event["photon_count"])
    print("Число попаданий на фотокатоды:", int(event["hits"].sum()))
    print("Фотоэлектроны по ФЭУ:", event["photoelectrons"].tolist())
    print("Полный заряд Q, фотоэлектроны:", event["Q"])
    print(f"Ожидаемое число попаданий: {event['photon_count'] * p:.3f}")
    print(f"Ожидаемый заряд: {mean_Q:.3f}")
    print(f"Стандартное отклонение заряда: {sigma_Q:.3f}")
    print(f"Отклонение результата от среднего: {(event['Q'] - mean_Q) / sigma_Q:.3f} sigma")
    print(f"Формула из условия без множителя 1/4: {formula_in_statement:.3f}")
