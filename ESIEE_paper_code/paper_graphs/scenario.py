import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Rectangle, Circle

# ==========================================================
# Load icons (must be in the same folder as this script)
# ==========================================================

drone_img = mpimg.imread("drone.png")
person_img = mpimg.imread("person.png")
station_img = mpimg.imread("station.png")

# ==========================================================
# Helper function to place icons
# ==========================================================

def add_icon(ax, img, x, y, zoom=0.12):
    ab = AnnotationBbox(
        OffsetImage(img, zoom=zoom),
        (x, y),
        frameon=False
    )
    ax.add_artist(ab)

# ==========================================================
# Create figure
# ==========================================================

fig, ax = plt.subplots(figsize=(12, 7))

# ==========================================================
# Disaster area
# ==========================================================

disaster_area = Rectangle(
    (0, 0),
    8,
    6,
    facecolor="whitesmoke",
    edgecolor="black",
    linewidth=2
)

ax.add_patch(disaster_area)

ax.text(
    4,
    6.2,
    "Disaster Area to cover",
    ha="center",
    fontsize=16,
    fontweight="bold"
)

# ==========================================================
# Victim groups
# ==========================================================

victim_groups = [
    (1.5, 1.5),
    (6.5, 2.0)
]

for x, y in victim_groups:

    coverage_zone = Circle(
        (x, y),
        0.8,
        fill=False,
        linestyle="--",
        linewidth=2,
        color="red"
    )

    ax.add_patch(coverage_zone)

    add_icon(ax, person_img, x, y, zoom=0.1)

    ax.text(
        x,
        y - 1.0,
        "Victim Group",
        ha="center",
        fontsize=10
    )

# ==========================================================
# UAV locations
# ==========================================================

uavs = {
    "UAV1": (2.0, 4.5),
    "UAV2": (4.0, 5.0),
    "UAV3": (6.0, 4.5),
    "UAV4": (3.0, 3.0),
    "UAV5": (5.0, 3.0)
}

# ==========================================================
# Communication links
# ==========================================================

links = [
    ("UAV1", "UAV2"),
    ("UAV2", "UAV3"),
    ("UAV1", "UAV4"),
    ("UAV2", "UAV4"),
    ("UAV2", "UAV5"),
    ("UAV3", "UAV5"),
    ("UAV4", "UAV5")
]

for a, b in links:

    x1, y1 = uavs[a]
    x2, y2 = uavs[b]

    ax.plot(
        [x1, x2],
        [y1, y2],
        linestyle="--",
        linewidth=2,
        color="dodgerblue"
    )

# ==========================================================
# Draw UAV icons
# ==========================================================

for name, (x, y) in uavs.items():

    add_icon(ax, drone_img, x, y, zoom=0.12)

    ax.text(
        x,
        y - 0.45,
        name,
        ha="center",
        fontsize=9
    )

# ==========================================================
# Base Station
# ==========================================================

add_icon(ax, station_img, 10.5, 3.0, zoom=0.15)

ax.text(
    10.5,
    1.8,
    "\n Base Station",
    ha="center",
    fontsize=12,
    fontweight="bold"
)

# Link from UAV network to command center

ax.plot(
    [6.0, 10.5],
    [4.5, 3.0],
    linestyle="--",
    linewidth=2.5,
    color="red"
)

# ==========================================================
# Final formatting
# ==========================================================

ax.set_xlim(-1, 12)
ax.set_ylim(-1, 7)

ax.set_aspect('equal')
ax.axis("off")

plt.tight_layout()
plt.savefig(
    "scenario.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()