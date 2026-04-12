import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

# ============================================================
# CONFIG
# ============================================================
IMAGE_PATH = "skin_example.jpg"   # <-- put your image filename here

N_AGENTS    = 5000    # number of agents
MAX_ITER    = 15      # SDS iterations
PATCH_SIZE  = 9       # local window size (odd number)
JITTER      = 4       # how far recruited agents can move from an active one
DARK_PCT    = 20      # darkest X% of pixels treated as "suspicious"

RANDOM_SEED = 42      # for reproducibility (optional)

# ============================================================
# SET SEED (optional but helpful for debugging)
# ============================================================
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ============================================================
# LOAD IMAGE AS GRAYSCALE ARRAY
# ============================================================
img_pil = Image.open(IMAGE_PATH).convert("L")  # L = grayscale
img = np.array(img_pil)
H, W = img.shape
print(f"Loaded image: {W} x {H}")

# ============================================================
# THRESHOLD FOR "DARK" AREAS (LESION)
#    We consider the darkest DARK_PCT% pixels as suspicious
# ============================================================
dark_threshold = np.percentile(img, DARK_PCT)
print(f"Dark threshold (percentile {DARK_PCT}): {dark_threshold:.2f}")

# ============================================================
# INITIALISE AGENTS
#   hypo_x[i], hypo_y[i] = hypothesis location of agent i
#   status[i] = True (active) / False (inactive)
# ============================================================
hypo_x = np.random.randint(0, W, size=N_AGENTS)
hypo_y = np.random.randint(0, H, size=N_AGENTS)
status = np.zeros(N_AGENTS, dtype=bool)

# ============================================================
# FITNESS FUNCTION: LOCAL MEAN BRIGHTNESS
#   Lower brightness = darker = more suspicious
# ============================================================
def local_brightness(x, y):
    half = PATCH_SIZE // 2
    x1 = max(0, x - half)
    x2 = min(W, x + half + 1)
    y1 = max(0, y - half)
    y2 = min(H, y + half + 1)
    patch = img[y1:y2, x1:x2]
    return float(np.mean(patch))

# ============================================================
# SDS MAIN LOOP
# ============================================================
activity_curve = []   # fraction of active agents per iteration

for itr in range(MAX_ITER):
    active_count = 0

    # ------------- TEST PHASE -------------
    for i in range(N_AGENTS):
        x, y = hypo_x[i], hypo_y[i]
        bri = local_brightness(x, y)

        # Active if patch is DARK enough (lesion-like)
        if bri < dark_threshold:
            status[i] = True
            active_count += 1
        else:
            status[i] = False

    frac_active = active_count / N_AGENTS
    activity_curve.append(frac_active)
    print(f"Iter {itr:02d}: active agents = {frac_active*100:.2f}%")

    # ------------- DIFFUSION PHASE -------------
    for i in range(N_AGENTS):
        if not status[i]:        # INACTIVE AGENT
            rand_idx = np.random.randint(N_AGENTS)

            if status[rand_idx]:  # recruit from ACTIVE
                # copy hypothesis with local jitter
                nx = hypo_x[rand_idx] + np.random.randint(-JITTER, JITTER + 1)
                ny = hypo_y[rand_idx] + np.random.randint(-JITTER, JITTER + 1)
                hypo_x[i] = int(np.clip(nx, 0, W - 1))
                hypo_y[i] = int(np.clip(ny, 0, H - 1))
            else:
                # random reinitialisation somewhere in the image
                hypo_x[i] = np.random.randint(0, W)
                hypo_y[i] = np.random.randint(0, H)

        else:                    # ACTIVE AGENT
            # small local move to keep exploring around lesion
            nx = hypo_x[i] + np.random.randint(-1, 2)
            ny = hypo_y[i] + np.random.randint(-1, 2)
            nx = int(np.clip(nx, 0, W - 1))
            ny = int(np.clip(ny, 0, H - 1))

            # re-test at new position (can become inactive)
            bri = local_brightness(nx, ny)
            status[i] = (bri < dark_threshold)
            hypo_x[i], hypo_y[i] = nx, ny

# ============================================================
# VISUALISATION: FINAL CLUSTERING
# ============================================================
plt.figure(figsize=(6, 6))
plt.imshow(img, cmap="gray")
plt.scatter(hypo_x[status],  hypo_y[status],  s=2, label="Active")
plt.scatter(hypo_x[~status], hypo_y[~status], s=1, alpha=0.2, label="Inactive")
plt.title("SDS Agents Clustering Over Dark Lesion")
plt.axis("off")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# VISUALISATION: ACTIVITY CURVE
# ============================================================
plt.figure()
plt.plot(np.arange(MAX_ITER), np.array(activity_curve) * 100, marker="o")
plt.xlabel("Iteration")
plt.ylabel("Active agents (%)")
plt.title("SDS Activity Curve")
plt.grid(True)
plt.tight_layout()
plt.show()
