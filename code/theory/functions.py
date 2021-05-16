def get_c_x_range(n, d, r):
    c_y_min = r ** 2 / (r + d)
    c_x_max = (r ** 2 - c_y_min ** 2) ** 0.5
    return map(lambda t: t * (c_x_max / n), range(-n, n + 1))


def get_c_z_range(m, h):
    return map(lambda t: t * (h * 0.8 / m), range(0, m))
