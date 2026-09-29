"""Animate a 1D Burgers solution evolving in time (shock formation).

Takes one initial condition u(x, 0) from the dataset and marches it to
u(x, 1) with the same pseudo-spectral integrator used to build the data,
capturing snapshots along the way. The resulting GIF shows the smooth
initial field steepening into a shock, which is the map the FNO learns to
approximate in a single step.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.solvers.burgers import solve_burgers_if


def make_evolution(u0, nu=0.01, t_final=1.0, n_frames=60, dt=1e-3):
    """March u0 (1, grid) to t_final, capturing n_frames+1 snapshots."""
    seg = t_final / n_frames
    frames = [u0.copy()]
    times = [0.0]
    u = u0.copy()
    for i in range(n_frames):
        u = solve_burgers_if(u, nu=nu, t_final=seg, dt=dt)
        frames.append(u.copy())
        times.append((i + 1) * seg)
    return np.array(frames)[:, 0, :], np.array(times)   # (n_frames+1, grid), (n_frames+1,)


def animate(frames, times, path, fps=15):
    grid = frames.shape[1]
    x = np.linspace(0.0, 1.0, grid, endpoint=False)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(x, frames[0], color="0.7", lw=1.3, ls="--", label="u(x, 0)")
    (line,) = ax.plot(x, frames[0], color="C0", lw=2.2, label="u(x, t)")
    pad = 0.1 * (frames.max() - frames.min())
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(frames.min() - pad, frames.max() + pad)
    ax.set_xlabel("x"); ax.set_ylabel("u")
    ax.legend(loc="upper right")
    title = ax.set_title("Burgers' evolution,  t = 0.000")
    fig.tight_layout()

    def update(i):
        line.set_ydata(frames[i])
        title.set_text(f"Burgers' evolution,  t = {times[i]:.3f}")
        return line, title

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def main():
    data = np.load("data/burgers.npz")
    u0_all = data["u0_test_256"].astype(np.float64)
    uT_all = data["uT_test_256"].astype(np.float64)

    # pick the test sample with the steepest final gradient (most dramatic shock)
    steepness = np.abs(np.diff(uT_all, axis=1)).max(axis=1)
    idx = int(steepness.argmax())

    u0 = u0_all[idx:idx + 1]
    frames, times = make_evolution(u0, nu=0.01, t_final=1.0, n_frames=60)

    # sanity check: the last frame should match the stored u(x, 1)
    rel = np.linalg.norm(frames[-1] - uT_all[idx]) / np.linalg.norm(uT_all[idx])
    print(f"test sample {idx} | final-frame vs stored uT rel-L2 = {rel:.2e}")

    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", "burgers_evolution.gif")
    animate(frames, times, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
