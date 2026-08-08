# Frank-Wolfe Algorithm Animation

This repository visualizes the **Frank-Wolfe algorithm** (also known as the Conditional Gradient Method) for two-dimensional convex optimization problems. The Python script computes the iterations numerically and displays every step as a 3D animation—from linearizing the objective function to performing a line search and obtaining the next iterate.

## What does the animation show?

The algorithm solves optimization problems of the form

\[
\min_{x \in D} f(x),
\]

where \(f:\mathbb{R}^2\rightarrow\mathbb{R}\) is a convex objective function and \(D\) is a compact, convex polygon.

For each iteration, the animation shows:

1. the current point \(x_k\),
2. the tangent plane of \(f\) at \(x_k\),
3. the solution \(s_k\) of the linearized subproblem,
4. the Frank-Wolfe direction \(d_k=s_k-x_k\),
5. the numerical line search along the segment between \(x_k\) and \(s_k\),
6. the new point \(x_{k+1}\).

The Frank-Wolfe gap is used as the stopping criterion:

\[
g_{\mathrm{FW}}(x_k)=\nabla f(x_k)^\top(x_k-s_k).
\]

## Requirements

- Python 3.10 or newer
- NumPy
- Matplotlib
- Pillow (for GIF export)

## Installation

Clone the repository and enter the project directory:

```bash
git clone <REPOSITORY-URL>
cd OR
```

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy matplotlib pillow
```

On Windows, activate the virtual environment with `.venv\Scripts\activate`.

## Usage

```bash
cd Animation_for_the_Franke-Wolfe-Algorithm
python franke_wolfe.py
```

The script prints a summary of the calculated iterations to the terminal, opens the animation in a Matplotlib window, and saves `franke_wolfe_animation.gif` in the current directory by default.

## Configuring your own optimization problem

The main settings are located near the beginning of [`franke_wolfe.py`](Animation_for_the_Franke-Wolfe-Algorithm/franke_wolfe.py):

```python
DOMAIN = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
])

START_POINT = np.array([0.2, 0.32])


def objective(x1, x2):
    return 6 * (x2 - 0.5) ** 2 + 6
```

| Setting | Description |
| --- | --- |
| `DOMAIN` | Vertices of the feasible convex polygon in boundary order |
| `START_POINT` | Starting point inside the polygon; use `None` to select the centroid of the vertices automatically |
| `objective` | Objective function \(f(x_1,x_2)\) |
| `TOLERANCE` | Stopping tolerance for the Frank-Wolfe gap |
| `MAX_ITERATIONS` | Maximum number of iterations |
| `SAVE_ANIMATION` | Enables or disables GIF export |
| `ANIMATION_FILE` | File name of the exported GIF |

`DOMAIN` must contain at least three finite points and describe a convex polygon. The vertices must be listed along the boundary in either clockwise or counterclockwise order. The Frank-Wolfe algorithm also assumes that the objective function is convex.

Using NumPy operations in the objective function improves rendering performance, but regular scalar Python functions are supported as well. If no analytical gradient is supplied, the program approximates it automatically using central finite differences.

## Programmatic usage

The animation can be created without displaying a window or exporting a GIF:

```python
from franke_wolfe import run_animation

animation = run_animation(save=False, show=False)
```

Define a custom problem with `FrankWolfeProblem`:

```python
import numpy as np

from franke_wolfe import FrankWolfeProblem, run_animation


def objective(x1, x2):
    return (x1 - 0.3) ** 2 + 2 * (x2 - 0.6) ** 2


problem = FrankWolfeProblem(
    objective=objective,
    domain=np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ]),
    start=np.array([0.8, 0.2]),
)

run_animation(problem=problem, tolerance=1e-3)
```

An analytical gradient can optionally be passed through the `gradient` argument when creating a `FrankWolfeProblem`.

## Implementation details

- The linear Frank-Wolfe subproblem is solved by evaluating the vertices of the polygon.
- The step size is determined using a derivative-free golden-section search on \([0,1]\).
- An additional reference point is calculated to mark the estimated global minimum in the visualization.
- The GIF resolution is selected automatically to remain below the configured frame-pixel limit.

## Project structure

```text
.
├── README.md
└── Animation_for_the_Franke-Wolfe-Algorithm/
    └── franke_wolfe.py
```
