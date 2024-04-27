from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle
import pathlib
import math

from absl import app, flags


def is_angle_in_range(angle, min_angle, max_angle):
    """
    Check if angle is in the range [min_angle, max_angle].
    Angles are expected to be in radians.
    """
    # Normalize the angles to the range [0, 2*pi)
    angle = angle % (2 * math.pi)
    min_angle = min_angle % (2 * math.pi)
    max_angle = max_angle % (2 * math.pi)

    if min_angle <= max_angle:
        # The range does not cross the 0/2*pi line
        return min_angle <= angle <= max_angle
    else:
        # The range crosses the 0/2*pi line
        return angle >= min_angle or angle <= max_angle


def compute_tangent_line_angles(lidar_pos, object_pos, radius):
    """
    Computing the tangent lines' angles from lidar
    """
    ax, ay = lidar_pos
    x1, y1 = object_pos
    # compute the distance from a to the center of the circle
    d = math.sqrt((x1 - ax)**2 + (y1 - ay)**2)

    # compute the angle from a to the center of the circle
    alpha = math.atan2(y1 - ay, x1 - ax)

    # compute the angle between the line from a to the center of the circle and the tangent lines
    # print(f'd: {d}, radius: {radius}, {radius / d}')
    assert d > radius, "D shoud be greater than radius"
    theta = math.asin(radius / d)

    # compute the angles of the tangent lines
    angle1 = alpha - theta
    angle2 = alpha + theta
    return d, alpha, angle1, angle2


def plot_objects(axe, row, lidar_pos, radius, arena_size):
    ax, ay = lidar_pos
    observed_objects = row['observation']
    tangent_pos = [compute_tangent_line_angles(
        lidar_pos, observed_object[:2], radius) for observed_object in observed_objects]
    tangent_start_ends = [(ax + d * np.cos(angle1), ay + d * np.sin(angle1), ax + d * np.cos(
        angle2), ay + d * np.sin(angle2)) for (d, _, angle1, angle2) in tangent_pos]
    d = arena_size * 1.42
    tangent_end_ends = [(ax + d * np.cos(angle1), ay + d * np.sin(angle1), ax + d * np.cos(
        angle2), ay + d * np.sin(angle2)) for (_, _, angle1, angle2) in tangent_pos]

    # Plot all objects
    for obj in row['all_objects']:
        circle = Circle((obj[0], obj[1]), radius, fill=False, color='r')
        axe.add_patch(circle)

    # Plot observable objects
    for obj in row['observation']:
        circle = Circle((obj[0], obj[1]), radius, fill=True, color='b')
        axe.add_patch(circle)

    # Plot tangent lines
    for (starts, ends) in zip(tangent_start_ends, tangent_end_ends):
        x_start1, y_start1, x_start2, y_start2 = starts
        x_end1, y_end1, x_end2, y_end2 = ends
        axe.plot([x_start1, x_end1], [y_start1, y_end1], 'g-')
        axe.plot([x_start2, x_end2], [y_start2, y_end2], 'g-')

    # Plot lidar
    axe.plot(lidar_pos[0], lidar_pos[1], 'ro')  # red circle for lidar
    axe.set_xlim(0, arena_size)
    axe.set_ylim(0, arena_size)

    # Set the aspect of the plot to be equal, so the circles look like circles
    axe.set_aspect('equal')
    # Add a grid
    axe.grid(True)


def plot_n_steps(df, lidar_pos, radius, arena_size, fig_path):
    fig, axe = plt.subplots(2, 5, figsize=(20, 10))

    for i, ax in enumerate(axe.flatten()):
        plot_objects(ax, df.loc[i], lidar_pos, radius, arena_size)
        ax.set_title(f'Occlusion at Time step {i}')

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.show()


def animate_objects(df, lidar_pos, radius, num_objects, xrange, yrange):
    # Create a new figure
    fig, ax = plt.subplots()

    # Set the aspect of the plot to be equal, so the circles look like circles
    ax.set_aspect('equal')

    # Set the x and y limits
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)

    # Initialize the circles
    all_objects_circles = [
        Circle((0, 0), radius, fill=False, color='r') for _ in range(num_objects)]
    observable_objects_circles = [
        Circle((0, 0), radius, fill=True, color='b') for _ in range(num_objects)]
    for circle in all_objects_circles + observable_objects_circles:
        ax.add_patch(circle)

    # Plot lidar
    ax.plot(lidar_pos[0], lidar_pos[1], 'ro')  # red circle for lidar

    # Animation update function
    def update(frame):
        # Update all objects circles
        for circle, obj in zip(all_objects_circles, df.iloc[frame]['all_objects']):
            circle.center = obj[0], obj[1]

        # Update observable objects circles
        for circle, obj in zip(observable_objects_circles, df.iloc[frame]['observation']):
            circle.center = obj[0], obj[1]

    # Create animation
    anim = FuncAnimation(fig, update, frames=range(len(df)), interval=200)

    # Show the plot
    plt.show()


def does_line_intersect_circle(start, angle, circle_center, arena_size, radius):
    # Calculate the end point of the line
    end = start + np.array([np.cos(angle), np.sin(angle)]) * arena_size

    # Calculate the distance from the circle center to the line
    d = np.abs(np.cross(end - start, start - circle_center)) / \
        np.linalg.norm(end - start)

    return d <= radius


def run_simulation(num_steps, objects, lidar_pos, arena_size, radius):
    df = pd.DataFrame(
        columns=["ts", "observation", "all_objects"], index=range(num_steps))
    tangent_pos = []

    # Simulation loop
    for t in range(num_steps):
        observable_objects, all_objects = [], []
        # Move objects
        objects[:, 0] += np.cos(objects[:, 2]) * \
            objects[:, 3]  # x += cos(angle) * speed
        objects[:, 1] += np.sin(objects[:, 2]) * \
            objects[:, 3]  # y += sin(angle) * speed
        # Change direction if objects reach the border or the lidar area
        objects[:, 2] += np.where((objects[:, 0] <= 0) | (objects[:, 0] >= arena_size) | (
            objects[:, 1] <= 0) | (objects[:, 1] >= arena_size), np.pi, 0)
        objects[:, 2] += np.where((np.sqrt((objects[:, 0] - lidar_pos[0])
                                  ** 2 + (objects[:, 1] - lidar_pos[1])**2)) <= radius, np.pi, 0)
        tangent_pos.clear()
        tangent_pos = [compute_tangent_line_angles(
            lidar_pos, obj[:2], radius) for obj in objects]
        # print(f'Time step: {t}, tangent_pos: {tangent_pos}')

        # Check for occlusion
        for i, obj in enumerate(objects):
            # Calculate the distance and angle to the object
            distance_to_object = np.hypot(
                obj[0] - lidar_pos[0], obj[1] - lidar_pos[1])
            angle_to_object = np.arctan2(
                obj[1] - lidar_pos[1], obj[0] - lidar_pos[0])
            assert math.isclose(distance_to_object, tangent_pos[i][0])
            assert math.isclose(angle_to_object, tangent_pos[i][1])

            # Check for intersection with other objects
            occluded = False
            for j, other_obj in enumerate(objects):
                if i != j:
                    distance_to_other = tangent_pos[j][0]
                    angle_to_other = tangent_pos[j][1]
                    tangent_angle1_other, tangent_angle2_other = tangent_pos[j][2], tangent_pos[j][3]

                    # If the other object is closer and in the same direction, the current object is occluded
                    if distance_to_other <= distance_to_object and is_angle_in_range(angle_to_object,
                                                                                     tangent_angle1_other, tangent_angle2_other):
                        occluded = True
                        break

            if not occluded:
                observable_objects.append(objects[i, :2].tolist())
            all_objects.append(objects[i, :2].tolist())
        df.iloc[t] = [t, observable_objects, all_objects]
    return df


FLAGS = flags.FLAGS
flags.DEFINE_integer("num_steps", 20, "Number of simulation steps")
flags.DEFINE_integer("num_objects", 6, "Number of objects")
flags.DEFINE_integer("arena_size", 10, "Size of the arena")
flags.DEFINE_float("radius", 0.25, "Radius of the objects")
flags.DEFINE_string("option", "plot", 'plot or animate')


def main(argv):
    del argv
    # Constants
    arena_size = FLAGS.arena_size
    num_objects = FLAGS.num_objects
    radius = FLAGS.radius
    num_steps = FLAGS.num_steps
    lidar_pos = np.array([arena_size / 2, 0])

    # Initialize objects
    objects = np.zeros((num_objects, 4))
    for i in range(num_objects):
        while True:
            x, y = np.random.rand(2) * arena_size  # x, y
            if np.sqrt((x - lidar_pos[0])**2 + (y - lidar_pos[1])**2) > radius:
                objects[i, :2] = x, y
                break
    # objects[:, :2] = np.random.rand(num_objects, 2) * arena_size  # x, y
    objects[:, 2] = np.random.rand(
        num_objects) * 2 * np.pi  # angle in [0, 2pi]
    objects[:, 3] = np.random.rand(num_objects) * 2 - 1  # speed in [-1, 1]

    data = run_simulation(num_steps, objects, lidar_pos, arena_size, radius)
    data_path = pathlib.Path("data")
    data_path.mkdir(exist_ok=True)
    data.to_csv(data_path / "data.csv", index=False)

    # plot_objects(data, lidar_pos, radius, arena_size)
    if FLAGS.option == 'plot':
        plot_n_steps(data, lidar_pos, radius, arena_size,
                     data_path / f"occlusion-{num_steps}.png")
    elif FLAGS.option == 'animate':
        animate_objects(data, lidar_pos, radius, num_objects,
                        (0, arena_size), (0, arena_size))


if __name__ == "__main__":
    app.run(main)
