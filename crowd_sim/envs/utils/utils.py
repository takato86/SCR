import numpy as np


def point_to_segment_dist(x1, y1, x2, y2, x3, y3):
    """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)

    """
    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:
        return np.linalg.norm((x3-x1, y3-y1))

    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py

    return np.linalg.norm((x - x3, y-y3))


def rotate_coordinate(x, y, theta):
    """rotate coordinates and return the points in the rotated one.

    Args:
        x ([float]): [description]
        y ([float]): [description]
        theta ([float]): [description]
    """
    rot_x = x * np.cos(theta) + y * np.sin(theta)
    rot_y = -x * np.sin(theta) + y * np.cos(theta)
    return rot_x, rot_y
