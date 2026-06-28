import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import csv
import math


# ----------------------
# CONFIG
# ----------------------
IMAGE_PATH = "skin_example.jpg"   # <-- put your image filename here

N_AGENTS    = 5000    # number of agents
MAX_ITER    = 15      # SDS iterations
PATCH_SIZE  = 9       # local window size (odd number)
JITTER      = 4       # how far recruited agents can move from an active one
DARK_PCT    = 20      # darkest X% of pixels treated as "suspicious"

RANDOM_SEED = 42

# Sphere mapping scales (tune if needed)
SCALE_BRI = 25.0    # scale for brightness difference (image 0-255)
SCALE_STD = 20.0    # scale for patch std difference

TOP_K = 100         # how many best agents to overlay on image
OUT_CSV = "sds_sphere_per_agent_final.csv"

# ----------------------
# SET SEED
# ----------------------
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ----------------------
# LOAD IMAGE
# ----------------------
img_pil = Image.open(IMAGE_PATH).convert("L")
img = np.array(img_pil)
H, W = img.shape
print(f"Loaded image: {W} x {H}")

# ----------------------
# DARK THRESHOLD
# ----------------------
dark_threshold = np.percentile(img, DARK_PCT)
print(f"Dark threshold (percentile {DARK_PCT}): {dark_threshold:.3f}")

# ----------------------
# INITIALISE AGENTS
# ----------------------
hypo_x = np.random.randint(0, W, size=N_AGENTS)
hypo_y = np.random.randint(0, H, size=N_AGENTS)
status = np.zeros(N_AGENTS, dtype=bool)

# ----------------------
# PATCH FUNCTIONS
# ----------------------
half = PATCH_SIZE // 2

def get_patch_coords(x, y):
    x1 = max(0, x - half)
    x2 = min(W, x + half + 1)
    y1 = max(0, y - half)
    y2 = min(H, y + half + 1)
    return x1, x2, y1, y2

def patch_stats_at(x, y):
    x1, x2, y1, y2 = get_patch_coords(x, y)
    patch = img[y1:y2, x1:x2]
    mean = float(np.mean(patch)) if patch.size else 0.0
    std = float(np.std(patch)) if patch.size else 0.0
    return mean, std

# ----------------------
# BASELINE PATCH STD (median)
#   estimate typical patch std across image to use as a target
# ----------------------
def estimate_baseline_patch_std(samples=2000):
    pts = []
    for _ in range(min(samples, W*H)):
        x = np.random.randint(0, W)
        y = np.random.randint(0, H)
        _, std = patch_stats_at(x, y)
        pts.append(std)
    return float(np.median(pts)) if pts else 0.0

BASE_STD_MED = estimate_baseline_patch_std(samples=3000)
print(f"Estimated baseline patch std (median): {BASE_STD_MED:.3f}")

# ----------------------
# FITNESS (Sphere) mapping for an agent at (x,y)
# ----------------------
EPS = 1e-9
def sphere_for_agent(x, y):
    mean, std = patch_stats_at(x, y)
    x1 = (mean - dark_threshold) / float(SCALE_BRI)
    x2 = (std  - BASE_STD_MED)  / float(SCALE_STD)
    sphere = float(x1*x1 + x2*x2)
    return sphere, mean, std, (x1, x2)

# ----------------------
# STORAGE for results
# ----------------------
activity_curve = []   # fraction of active agents per iteration
sphere_stats_per_iter = []  # list of dicts: mean,median,var,std,min,max
# We'll also keep final per-agent data
final_agent_data = np.zeros((N_AGENTS,), dtype=[('x','i4'),('y','i4'),('status','?'),
                                                ('sphere','f8'),('mean','f8'),('std','f8')])

# ----------------------
# SDS MAIN LOOP (augmented to compute sphere values each iter)
# ----------------------
for itr in range(MAX_ITER):
    active_count = 0

    # TEST PHASE
    # evaluate brightness and activation
    for i in range(N_AGENTS):
        x, y = int(hypo_x[i]), int(hypo_y[i])
        bri, _ = patch_stats_at(x, y)
        if bri < dark_threshold:
            status[i] = True
            active_count += 1
        else:
            status[i] = False

    frac_active = active_count / N_AGENTS
    activity_curve.append(frac_active)
    print(f"Iter {itr:02d}: active agents = {frac_active*100:.2f}%")

    # compute sphere values for all agents at this iteration and record stats
    spheres = np.empty(N_AGENTS, dtype=float)
    means = np.empty(N_AGENTS, dtype=float)
    stds = np.empty(N_AGENTS, dtype=float)
    for i in range(N_AGENTS):
        x, y = int(hypo_x[i]), int(hypo_y[i])
        s, m, st, _ = sphere_for_agent(x, y)
        spheres[i] = s
        means[i] = m
        stds[i] = st

    # summary stats
    stats = {
        'mean': float(np.mean(spheres)),
        'median': float(np.median(spheres)),
        'var': float(np.var(spheres)),
        'std': float(np.std(spheres)),
        'min': float(np.min(spheres)),
        'max': float(np.max(spheres))
    }
    sphere_stats_per_iter.append(stats)

    # DIFFUSION PHASE
    for i in range(N_AGENTS):
        if not status[i]:        # INACTIVE AGENT
            rand_idx = np.random.randint(N_AGENTS)
            if status[rand_idx]:  # recruit from ACTIVE
                nx = hypo_x[rand_idx] + np.random.randint(-JITTER, JITTER + 1)
                ny = hypo_y[rand_idx] + np.random.randint(-JITTER, JITTER + 1)
                hypo_x[i] = int(np.clip(nx, 0, W - 1))
                hypo_y[i] = int(np.clip(ny, 0, H - 1))
            else:
                hypo_x[i] = np.random.randint(0, W)
                hypo_y[i] = np.random.randint(0, H)
        else:                    # ACTIVE AGENT
            nx = hypo_x[i] + np.random.randint(-1, 2)
            ny = hypo_y[i] + np.random.randint(-1, 2)
            nx = int(np.clip(nx, 0, W - 1))
            ny = int(np.clip(ny, 0, H - 1))
            bri = patch_stats_at(nx, ny)[0]
            status[i] = (bri < dark_threshold)
            hypo_x[i], hypo_y[i] = nx, ny

# ----------------------
# After run: compute final per-agent sphere values and save
# ----------------------
for i in range(N_AGENTS):
    x, y = int(hypo_x[i]), int(hypo_y[i])
    s, m, st, _ = sphere_for_agent(x, y)
    final_agent_data[i] = (x, y, bool(status[i]), float(s), float(m), float(st))

# write CSV with final per-agent data
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["agent_index","x","y","status","sphere","mean_patch","std_patch"])
    for i in range(N_AGENTS):
        row = final_agent_data[i]
        writer.writerow([int(i), int(row['x']), int(row['y']), bool(row['status']),
                         float(row['sphere']), float(row['mean']), float(row['std'])])
print(f"Saved final per-agent data to {OUT_CSV}")

# ----------------------
# Print final sphere summary (same as last iter stats)
# ----------------------
final_stats = sphere_stats_per_iter[-1]
print("\nFinal Sphere stats (agents):")
print(f" mean   = {final_stats['mean']:.6e}")
print(f" median = {final_stats['median']:.6e}")
print(f" var    = {final_stats['var']:.6e}")
print(f" std    = {final_stats['std']:.6e}")
print(f" min    = {final_stats['min']:.6e}")
print(f" max    = {final_stats['max']:.6e}")

# ----------------------
# PLOTTING
# ----------------------
# 1) Histogram of final Sphere values
plt.figure(figsize=(6,4))
plt.hist(final_agent_data['sphere'], bins=80, color='C0', alpha=0.8)
plt.xlabel("Sphere value (lower = closer to target)")
plt.ylabel("Number of agents")
plt.title("Histogram of final Sphere values across agents")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2) Overlay top-K best agents on image
best_idx = np.argsort(final_agent_data['sphere'])[:TOP_K]
plt.figure(figsize=(6,6))
plt.imshow(img, cmap="gray")
# plot all inactive agents faintly
inactive_idx = np.where(~final_agent_data['status'])[0]
plt.scatter(final_agent_data['x'][inactive_idx], final_agent_data['y'][inactive_idx],
            s=1, alpha=0.12, color='cyan', label='inactive')
# plot top-K best (lowest Sphere) in red
plt.scatter(final_agent_data['x'][best_idx], final_agent_data['y'][best_idx],
            s=8, color='red', marker='o', label=f'top {len(best_idx)} (best Sphere)')
plt.title("Top agents by Sphere fitness (red)")
plt.axis("off")
plt.legend()
plt.tight_layout()
plt.show()

# 3) Activity curve (was recorded)
plt.figure(figsize=(6,3))
iters = np.arange(len(activity_curve))
plt.plot(iters, np.array(activity_curve) * 100, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Active agents (%)")
plt.title("SDS Activity Curve")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4) Per-iteration Sphere statistics table (print)
print("\nPer-iteration Sphere statistics (mean, median, var, std, min, max):")
for i, s in enumerate(sphere_stats_per_iter):
    print(f"Iter {i:02d}: mean={s['mean']:.6e}, median={s['median']:.6e}, var={s['var']:.6e}, std={s['std']:.6e}, min={s['min']:.6e}, max={s['max']:.6e}")

# Save summary stats to disk as txt
summary_txt = "sds_sphere_summary.txt"
with open(summary_txt, "w") as f:
    f.write("Final Sphere stats (agents):\n")
    for k,v in final_stats.items():
        f.write(f"{k}: {v}\n")
    f.write("\nPer-iteration Sphere stats:\n")
    for i, s in enumerate(sphere_stats_per_iter):
        f.write(f"Iter {i}: {s}\n")
print(f"Saved summary to {summary_txt}")
