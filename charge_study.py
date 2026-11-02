"""Серии событий для сферического сцинтилляционного детектора."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.spatial import cKDTree

from detector import Detector
from source import Source


ENERGIES = (0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
POSITIONS = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
N_EVENTS = 1000
SEED = 20261028
BLUE, ORANGE = "#22618C", "#CF6A32"


def boundary_points(detector, position, directions):
    """Векторизованный выход лучей из строго внутренней точки на сферу."""
    position = np.asarray(position, dtype=float)
    directions = np.asarray(directions, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("position должна быть конечной тройкой координат.")
    if np.linalg.norm(position) >= detector.R:
        raise ValueError("В этой серии источник должен быть строго внутри сферы.")
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions должен иметь форму (число фотонов, 3).")
    if not np.isfinite(directions).all():
        raise ValueError("Направления должны быть конечными.")
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths == 0):
        raise ValueError("Направления должны быть ненулевыми.")
    directions = directions / lengths[:, None]
    b = directions @ position
    c = position @ position - detector.R**2
    root = np.sqrt(b**2 - c)
    t = root - b
    forward = b >= 0
    t[forward] = -c / (b[forward] + root[forward])
    return position + t[:, None] * directions


def simulate_series(detector, source, E, position, n_events=1000):
    """Возвращает заряды и числа попаданий для независимых событий."""
    tree = cKDTree(detector.pmt_positions)
    charges = np.empty(n_events, dtype=np.int64)
    hits = np.empty(n_events, dtype=np.int64)
    for event in range(n_events):
        directions = source.sample_directions(E)
        points = boundary_points(detector, position, directions)
        distances, _ = tree.query(points, distance_upper_bound=detector.r + 1e-9)
        hits[event] = np.count_nonzero(np.isfinite(distances))
        charges[event] = detector.sample_photoelectrons(size=int(hits[event])).sum()
    return charges, hits


def collection_probability(detector, position, n_radial=16, n_azimuth=64):
    """Численный интеграл телесного угла неперекрывающихся фотокатодов.

    С поверхности dS виден телесный угол
    dOmega = dot(q/R, q-position) / |q-position|**3 * dS.
    Интегрирование по каждому сферическому фотокатоду: квадратура Гаусса
    по cos(theta) и равномерная периодическая сетка по азимуту.
    Используется только для теоретического сравнения; MC от него не зависит.
    """
    position = np.asarray(position, dtype=float)
    if np.linalg.norm(position) >= detector.R:
        raise ValueError("Источник должен быть строго внутри сферы.")
    separations = np.linalg.norm(
        detector.pmt_positions[:, None] - detector.pmt_positions[None, :], axis=-1
    )
    np.fill_diagonal(separations, np.inf)
    if separations.min() <= 2 * detector.r:
        raise ValueError("Для этого интеграла нужны неперекрывающиеся фотокатоды.")

    nodes, weights = leggauss(n_radial)
    # cos(alpha) = 1 - r**2/(2R**2), так как r измеряется по хорде.
    cos_alpha = 1 - detector.r**2 / (2 * detector.R**2)
    mu = (1 + cos_alpha) / 2 + (1 - cos_alpha) / 2 * nodes
    weights = (1 - cos_alpha) / 2 * weights
    phi = np.arange(n_azimuth) * 2 * np.pi / n_azimuth
    probability = 0.0
    for normal in detector.pmt_positions / detector.R:
        reference = np.eye(3)[np.argmin(np.abs(normal))]
        u = np.cross(normal, reference)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        rim = np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v
        normals = mu[:, None, None] * normal + np.sqrt(1 - mu**2)[:, None, None] * rim
        rays = detector.R * normals - position
        integrand = np.sum(normals * rays, axis=-1) / np.linalg.norm(rays, axis=-1)**3
        probability += detector.R**2 / 2 * np.sum(weights * integrand.mean(axis=1))
    return float(probability)


def describe(charges, hits, E, z, probability, detector, source):
    n = len(charges)
    mean = float(charges.mean())
    std = float(charges.std(ddof=1))
    ratio = std / mean
    # Дельта-метод для ошибки sigma/mean с учётом третьего и четвёртого моментов.
    centered = charges - mean
    v = float(np.mean(centered**2))
    m3, m4 = float(np.mean(centered**3)), float(np.mean(centered**4))
    rel_var = ratio**2 / n * (v / mean**2 + (m4 - v**2) / (4 * v**2) - m3 / (mean * v))
    M = int(round(source.Y * E))
    expected_mean = M * probability * detector.a
    expected_std = np.sqrt(M * (probability * detector.a + probability * (1 - probability) * detector.a**2))
    return {
        "E_MeV": E, "z_m": z, "n_events": n,
        "photons_per_event": M, "mean_hits": float(hits.mean()),
        "mean_Q_pe": mean, "std_Q_pe": std, "se_mean_Q_pe": std / np.sqrt(n),
        "relative_width": ratio, "relative_width_percent": ratio * 100,
        "se_relative_width": float(np.sqrt(max(0.0, rel_var))),
        "p_geometric": probability, "expected_mean_Q_pe": expected_mean,
        "expected_std_Q_pe": float(expected_std),
        "expected_relative_width": float(expected_std / expected_mean),
    }


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E9EE", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def plot_distributions(outdir, records, samples):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.7))
    for ax, E in zip(axes.flat, ENERGIES):
        q0, q1 = samples[(0.0, E)][0], samples[(1.0, E)][0]
        lo, hi = min(q0.min(), q1.min()), max(q0.max(), q1.max())
        step = max(1, int(np.ceil((hi - lo + 1) / 32)))
        bins = np.arange(lo - 0.5, hi + step + 0.5, step)
        ax.hist(q0, bins=bins, color=BLUE, alpha=0.32, label="Центр", zorder=2)
        ax.hist(q1, bins=bins, color=ORANGE, histtype="step", linewidth=1.5, label="z = 1 м", zorder=3)
        r0 = next(r for r in records if r["E_MeV"] == E and r["z_m"] == 0)
        r1 = next(r for r in records if r["E_MeV"] == E and r["z_m"] == 1)
        ax.set_title(f"E = {E:g} МэВ", loc="left", fontsize=13, fontweight="bold", pad=10)
        ax.text(0.97, 0.96, f"Центр: μ = {r0['mean_Q_pe']:.2f}, σ = {r0['std_Q_pe']:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=BLUE)
        ax.text(0.97, 0.86, f"z = 1 м: μ = {r1['mean_Q_pe']:.2f}, σ = {r1['std_Q_pe']:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=ORANGE)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.30)
        ax.set_xlabel("Q, фотоэлектроны", fontsize=10)
        ax.set_ylabel("Число событий", fontsize=10)
        style_axes(ax)
    ax = axes.flat[-1]
    ax.axis("off")
    ax.plot([], [], color=BLUE, linewidth=8, alpha=0.45, label="Центр: (0, 0, 0) м")
    ax.plot([], [], color=ORANGE, linewidth=1.7, label="Смещение: (0, 0, 1) м")
    ax.legend(loc="upper left", frameon=False, fontsize=11, bbox_to_anchor=(0, 1.02))
    ax.text(0.02, 0.67, "1000 независимых событий\nдля каждой энергии и положения\n\nR = 2 м · N = 40 · r = 0,10 м\nY = 10 000 фотонов/МэВ · a = 0,5\n\nμ — средний заряд\nσ — стандартное отклонение",
            transform=ax.transAxes, fontsize=11, linespacing=1.6, va="top", color="#465363")
    fig.suptitle("Распределения полного собранного заряда", x=0.06, ha="left", fontsize=22, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.89, wspace=0.33, hspace=0.43)
    fig.savefig(outdir / "charge_distributions.png", dpi=180, facecolor="white")
    plt.close(fig)


def plot_trends(outdir, records, probabilities):
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 5.25))
    Egrid = np.linspace(0.18, 3.1, 400)
    p = probabilities[0.0]
    coefficient = np.sqrt((1 + 0.5 * (1 - p)) / (10000 * p * 0.5))
    axes[0].plot(Egrid, 100 * coefficient / np.sqrt(Egrid), color="#44505E", linestyle="--", linewidth=1.4,
                 label=r"Теория: $10{,}91\%/\sqrt{E/\mathrm{МэВ}}$")
    for z, color, marker, label in ((0.0, BLUE, "o", "Центр"), (1.0, ORANGE, "s", "z = 1 м")):
        rows = [r for r in records if r["z_m"] == z]
        axes[0].errorbar([r["E_MeV"] for r in rows], [r["relative_width_percent"] for r in rows],
                         yerr=[r["se_relative_width"] * 100 for r in rows],
                         fmt=marker, ms=5, color=color, capsize=3, label=label, zorder=3)
    axes[0].set_title("Относительная ширина", loc="left", fontweight="bold", fontsize=14, pad=12)
    axes[0].set_xlabel("Энергия, МэВ")
    axes[0].set_ylabel(r"$\sigma_Q / \langle Q\rangle$, %")
    axes[0].legend(frameon=False, fontsize=10)
    axes[0].set_xlim(0.1, 3.15)
    axes[0].set_ylim(bottom=0)

    differences, errors = [], []
    for E in ENERGIES:
        r0 = next(r for r in records if r["E_MeV"] == E and r["z_m"] == 0)
        r1 = next(r for r in records if r["E_MeV"] == E and r["z_m"] == 1)
        m0, m1 = r0["mean_Q_pe"], r1["mean_Q_pe"]
        differences.append(100 * (m1 / m0 - 1))
        errors.append(100 * np.sqrt((r1["se_mean_Q_pe"] / m0)**2 + (m1 * r0["se_mean_Q_pe"] / m0**2)**2))
    predicted = 100 * (probabilities[1.0] / probabilities[0.0] - 1)
    axes[1].axhline(0, color="#BBC3CD", linewidth=1)
    axes[1].axhline(predicted, color=ORANGE, linewidth=1.3, linestyle="--", label=f"Геометрия: {predicted:+.3f}%")
    axes[1].errorbar(ENERGIES, differences, yerr=errors, fmt="o", color=BLUE, capsize=4,
                    label="Разность средних, MC", zorder=3)
    axes[1].set_title("Изменение среднего при смещении", loc="left", fontweight="bold", fontsize=14, pad=12)
    axes[1].set_xlabel("Энергия, МэВ")
    axes[1].set_ylabel(r"$100\,(\langle Q\rangle_{z=1}/\langle Q\rangle_{z=0}-1)$, %")
    axes[1].set_xlim(0.1, 3.15)
    axes[1].legend(frameon=False, fontsize=10)
    for ax in axes:
        style_axes(ax)
    fig.suptitle("Энергетическая зависимость и влияние положения источника", x=0.07, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.07, 0.03, "По 1000 событий на точку. Погрешности — оценки 1σ; заряд измеряется в фотоэлектронах.", fontsize=10, color="#596575")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.18, wspace=0.3)
    fig.savefig(outdir / "charge_trends.png", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    outdir = Path(__file__).resolve().parent / "charge_results"
    outdir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    detector_geometry = Detector()
    probabilities = {}
    for position in POSITIONS:
        p1 = collection_probability(detector_geometry, position)
        p2 = collection_probability(detector_geometry, position, 32, 128)
        if not np.isclose(p1, p2, atol=1e-12, rtol=0):
            raise RuntimeError("Недостаточная точность квадратуры телесного угла.")
        probabilities[position[2]] = p2
    streams = iter(np.random.SeedSequence(SEED).spawn(2 * len(POSITIONS) * len(ENERGIES)))
    records, samples = [], {}
    for position in POSITIONS:
        z = position[2]
        for E in ENERGIES:
            source = Source(seed=next(streams))
            detector = Detector(seed=next(streams))
            charges, hits = simulate_series(detector, source, E, position, N_EVENTS)
            row = describe(charges, hits, E, z, probabilities[z], detector, source)
            records.append(row)
            samples[(z, E)] = (charges, hits)
            print(f"z={z:g} м, E={E:g} МэВ: mean={row['mean_Q_pe']:.4f}, sigma={row['std_Q_pe']:.4f}, width={row['relative_width_percent']:.3f}%", flush=True)

    with (outdir / "charge_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (outdir / "event_charges.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("z_m", "E_MeV", "event_index", "hits", "Q_pe"))
        for (z, E), (charges, hits) in samples.items():
            writer.writerows((z, E, i, int(h), int(q)) for i, (h, q) in enumerate(zip(hits, charges)))
    exponents = {}
    for z in probabilities:
        rows = [r for r in records if r["z_m"] == z]
        exponents[z] = float(np.polyfit(np.log(ENERGIES), np.log([r["relative_width"] for r in rows]), 1)[0])
    metadata = {"seed": SEED, "n_events_per_setting": N_EVENTS, "energies_MeV": ENERGIES,
                "p_geometric": probabilities, "relative_mean_shift_percent": 100 * (probabilities[1.0] / probabilities[0.0] - 1),
                "width_power_law_exponents": exponents, "statistics": records}
    (outdir / "study_results.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_distributions(outdir, records, samples)
    plot_trends(outdir, records, probabilities)
    print(json.dumps({k: v for k, v in metadata.items() if k != "statistics"}, ensure_ascii=False, indent=2))
    print("Готовые графики и таблицы:", outdir.name)


if __name__ == "__main__":
    main()
