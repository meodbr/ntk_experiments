import numpy as np
import matplotlib.pyplot as plt
# import umap

def unit_sphere(dim=2, steps=1000, seed=None):
    # Regular walk around the unit circle/sphere
    gamma = np.arange(steps) * 2 * np.pi / steps - np.pi
    if dim == 2:
        return gamma, np.column_stack([np.cos(gamma), np.sin(gamma)])
    else:
        rng = np.random.default_rng(seed)
        traj = rng.normal(size=(steps, dim))
        traj /= np.linalg.norm(traj, axis=1, keepdims=True)
        return gamma, traj

def random_walk_unit_sphere(
    dim=64,
    steps=1000,
    step_size=0.05,
    seed=None
):
    rng = np.random.default_rng(seed)

    # Random initial point on unit sphere
    x = rng.normal(size=dim)
    x /= np.linalg.norm(x)

    trajectory = np.empty((steps, dim))
    trajectory[0] = x

    for i in range(1, steps):
        # Random perturbation
        noise = rng.normal(size=dim)

        # Remove radial component so movement is tangent to sphere
        noise -= np.dot(noise, x) * x

        # Normalize tangent direction
        noise /= np.linalg.norm(noise)

        # Small move
        x = x + step_size * noise

        # Project back to unit sphere
        x /= np.linalg.norm(x)

        trajectory[i] = x

    return trajectory

def sphere_to_2d_projection(traj):
    rng = np.random.default_rng()

    # random orthonormal basis in R^64
    a = rng.normal(size=64)
    a /= np.linalg.norm(a)

    b = rng.normal(size=64)
    b -= np.dot(b, a) * a
    b /= np.linalg.norm(b)

    x = traj @ a
    y = traj @ b

    return np.column_stack([x, y])

def plot_sphere_path(projection):
    plt.figure(figsize=(8, 8))
    plt.plot(projection[:, 0], projection[:, 1], marker='o', markersize=2)
    plt.title('Random Walk on Unit Sphere (2D Projection)')
    plt.xlabel('Projection 1')
    plt.ylabel('Projection 2')
    plt.axis('equal')
    plt.grid()
    plt.show()

# def plot_trajectory(trajectory):
#     reducer = umap.UMAP(n_components=2, random_state=42)
#     embedding = reducer.fit_transform(trajectory)

#     plt.figure(figsize=(8, 6))
#     plt.scatter(embedding[:, 0], embedding[:, 1], c=np.arange(len(embedding)), cmap='viridis', s=5)
#     plt.colorbar(label='Step')
#     plt.title('Random Walk on Unit Sphere (UMAP Projection)')
#     plt.xlabel('UMAP Dimension 1')
#     plt.ylabel('UMAP Dimension 2')
#     plt.show()


if __name__ == "__main__":
    traj = random_walk_unit_sphere()

    print(traj.shape)      # (1000, 64)
    print(np.linalg.norm(traj[0]))   # ~1
    print(np.linalg.norm(traj[-1]))  # ~1
    plot_sphere_path(sphere_to_2d_projection(traj))