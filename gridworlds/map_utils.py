# Various helper functions for working with map layouts
# Copied from my spatial-cognition repo, not all of these are used here
# TODO: clean this up a little

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.misc
import time
import pickle
import json


def gaussian_1d(x, sigma, linspace):
    return np.exp(-((linspace - x) ** 2) / sigma / sigma)


def gaussian_2d(x, y, sigma, meshgrid):
    X, Y = meshgrid
    return np.exp(-((X - x) ** 2 + (Y - y) ** 2) / sigma / sigma)


def layout_to_array(layout):
    rows = layout.strip().split('\n')
    shape = (len(rows), len(rows[0]))

    arr = np.zeros(shape)

    for r in range(shape[0]):
        for c in range(shape[1]):
            arr[r, c] = 1 if rows[r][c] == '#' else 0

    return arr


def array_to_layout(arr):
    layout = ''

    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            if arr[r, c] == 1:
                layout += '#'
            else:
                layout += ' '
        layout += '\n'

    return layout


def layout_to_image(layout, res):
    arr = layout_to_array(layout)

    return scipy.misc.imresize(arr, size=(res, res)), arr


def array_to_image(arr, res):
    return scipy.misc.imresize(arr, size=(res, res))


def coord_to_pixel(coord, map_res=8, im_res=256):
    pixel = np.round(coord * im_res / map_res).astype(np.int)

    return pixel


def allo_occupancy(x, y, map_arr, pob, zoom_level=4):
    """
    Generate an occupancy map centered on the agent in allocentric coordinates
    x (float): x location of the agent in the map
    y (float): y location of the agent in the map
    map_arr (np.array): array representation of the map
    pob (int): side length of square subsection of the map to represent
    #res (int): resolution of the output. Output is a res X res numpy array
    zoom_level (int): determines resolution of the output. Expands each cell to be a zl by zl square
    """

    # Transform the map to the desired resolution
    arr_zoom = scipy.ndimage.zoom(map_arr, zoom_level, order=0)
    arr_len = len(arr_zoom)

    # Find closest center pixel based on x-y position
    xz = int(np.round(x * zoom_level))
    yz = int(np.round(y * zoom_level))

    # Slice out section of the map based on pob size (translated to new resolution)
    pobz = pob * zoom_level

    # padding = int((pobz - 1)/2)
    padding = int((pobz) / 2) + zoom_level
    arr_zoom_padded = np.ones((arr_len + padding * 2, arr_len + padding * 2))
    arr_zoom_padded[padding:-padding, padding:-padding] = arr_zoom

    # TODO: make sure this math is correct
    # pob_occupancy = arr_zoom_padded[xz:xz+pobz, yz:yz+pobz]
    pob_occupancy = arr_zoom_padded[zoom_level + xz:zoom_level + xz + pobz, zoom_level + yz:zoom_level + yz + pobz]

    # Return output map
    return pob_occupancy


def allo_occupancy_pre_zoomed(x, y, arr_zoom_padded, pob, zoom_level):
    """
    Same as above but assumes the zoomed array has been precalculated
    """
    # arr_len = len(arr_zoom)

    # Find closest center pixel based on x-y position
    xz = int(np.round(x * zoom_level))
    yz = int(np.round(y * zoom_level))

    # Slice out section of the map based on pob size (translated to new resolution)
    pobz = pob * zoom_level

    # padding = int((pobz - 1)/2)
    # arr_zoom_padded = np.ones((arr_len+padding*2, arr_len+padding*2))
    # arr_zoom_padded[padding:-padding, padding:-padding] = arr_zoom

    # TODO: make sure this math is correct
    # pob_occupancy = arr_zoom_padded[xz:xz+pobz, yz:yz+pobz]
    # pob_occupancy = arr_zoom_padded[zoom_level+xz:zoom_level+xz+pobz, zoom_level+yz:zoom_level+yz+pobz]
    # NOTE: x and y axes seem to be flipped from convention here
    pob_occupancy = arr_zoom_padded[zoom_level + yz:zoom_level + yz + pobz,
                    zoom_level + xz:zoom_level + xz + pobz]

    # Return output map
    return pob_occupancy


def allo_occupancy_gaussian(x, y, map_arr, pob, res, std=.5):
    """
    Convolve a gaussian across a layout array onto an output with a specific resolution
    """
    # Generate numpy array of all indices in one dimension of the explored_space
    xs = np.linspace(x - pob / 2., x + pob / 2., res, endpoint=True)
    ys = np.linspace(y - pob / 2., y + pob / 2., res, endpoint=True)

    # Generate mesh in two dimensions
    mg = np.meshgrid(xs, ys)

    output_arr = np.zeros((res, res))

    # TODO: optimize so it only scans pob+1 rather than the entire array
    for i in range(map_arr.shape[0]):
        for j in range(map_arr.shape[1]):
            if map_arr[i, j] == 1:
                output_arr += gaussian_2d(x=i, y=j, sigma=std, meshgrid=mg)

    return output_arr


def ego_occupancy(x, y, theta, map_arr, pob, zoom_level=4):
    """
    The same as allo_occupancy but rotated based on theta
    """
    pass


def ego_occupancy_from_sensors(sensor_dists,
                               pob,
                               fov=90,
                               zoom_level=4,
                               sigma=0.5):
    """
    Construct a partial occupancy map based on distance sensor measurements
    """
    return allo_occupancy_from_sensors(
        sensor_dists=sensor_dists,
        pob=pob,
        head_direction=0,
        fov=fov,
        zoom_level=zoom_level,
        sigma=sigma,
    )


# TODO: make this more efficient
def allo_occupancy_from_sensors(sensor_dists,
                                pob,
                                head_direction,
                                fov=90,
                                zoom_level=4,
                                sigma=0.5):
    """
    Construct a partial occupancy map based on distance sensor measurements
    Rotated by the head direction into an allocentric reference frame
    """
    fov_rad = fov * np.pi / 180.

    # Set up space for placing gaussians at obstacle detection points
    lin_domain = np.linspace(-pob / 2., pob / 2., pob * zoom_level)
    x_domain, y_domain = np.meshgrid(lin_domain, lin_domain)

    occ_array = np.zeros((pob * zoom_level, pob * zoom_level))

    ang_interval = np.pi / len(sensor_dists)
    start_ang = -fov_rad / 2. + head_direction

    for i, dist in enumerate(sensor_dists):
        x = dist * np.cos(start_ang + i * ang_interval)
        y = dist * np.sin(start_ang + i * ang_interval)
        occ_array += gaussian_2d(x, y, sigma=sigma, meshgrid=[x_domain, y_domain])

    return occ_array


def generate_sensor_readings(map_arr,
                             zoom_level=4,
                             n_sensors=30,
                             fov_rad=np.pi,
                             x=0,
                             y=0,
                             th=0,
                             max_sensor_dist=10,
                             debug_value=0,  # For debugging the function interactively
                             ):
    """
    Given a map, agent location in the map, number of sensors, field of view
    calculate the distance readings of each sensor to the nearest obstacle
    uses supersampling to find the approximate collision points
    """
    arr_zoom = scipy.ndimage.zoom(map_arr, zoom_level, order=0)
    dists = np.zeros((n_sensors,))

    angs = np.linspace(-fov_rad / 2. + th, fov_rad / 2. + th, n_sensors)

    for i, ang in enumerate(angs):
        dists[i] = get_collision_coord(arr_zoom, x * zoom_level, y * zoom_level, ang, max_sensor_dist * zoom_level,
                                       debug_value=debug_value) / zoom_level

    return dists


def get_collision_coord(map_array, x, y, th,
                        max_sensor_dist=10 * 4,
                        debug_value=0,
                        ):
    """
    Find the first occupied space given a start point and direction
    Meant for a zoomed in map_array
    """
    # Define the step sizes
    dx = np.cos(th)
    dy = np.sin(th)

    # Initialize to starting point
    cx = x
    cy = y

    for i in range(max_sensor_dist):
        # Move one unit in the direction of the sensor
        cx += dx
        cy += dy

        # If the cell is occupied, return the distance travelled to get there
        # TODO: make sure the rounding is correct
        """
        #if map_array[int(round(cx)), int(round(cy))] == 1:
        #if map_array[int(cx+.5), int(cy+.5)] == 1:
        #if map_array[int(round(cx+.5)), int(round(cy+.5))] == 1:
        if map_array[int(cx), int(cy)] == 1:
            return (i-1)
        """
        # debug value of 0 seems to be the best.
        # Misalignments in the visualization seem to be an artifact, biased differently in different sides of the svg, so likely some scaling thing, and the actual value should hopefully be correct
        if debug_value == 0:
            if map_array[int(cx), int(cy)] == 1:
                # return (i-1)
                return i
        elif debug_value == 1:
            if map_array[int(cx + .5), int(cy + .5)] == 1:
                return (i - 1)
        elif debug_value == 2:
            if map_array[int(round(cx)), int(round(cy))] == 1:
                return (i - 1)
        elif debug_value == 3:
            if map_array[int(round(cx + .5)), int(round(cy + .5))] == 1:
                return (i - 1)

    return max_sensor_dist


def test_map_scan():
    """
    Scans around a map and plots the allo view
    Test to make sure it makes sense
    """
    import matplotlib.pyplot as plt

    pob = 5
    zoom_level = 4
    pobz = pob * zoom_level

    # Pre-generate padded zoomed array for efficiency
    map_arr = layout_to_array(map_layouts[0])
    arr_zoom = scipy.ndimage.zoom(map_arr, zoom_level, order=0)

    arr_len = len(arr_zoom)
    padding = int((pobz) / 2) + zoom_level  # add zoom_level to padding to ensure everything is in there
    arr_zoom_padded = np.ones((arr_len + padding * 2, arr_len + padding * 2))
    arr_zoom_padded[padding:-padding, padding:-padding] = arr_zoom

    # Figures for occupancy plot (real and estimated)
    fig, ax_arr = plt.subplots(2)

    layout_im, layout_arr = layout_to_image(map_layouts[0], res=8 * zoom_level)

    print(layout_im)

    images = []

    images.append(ax_arr[0].imshow(np.random.random((pobz, pobz))))
    images.append(ax_arr[1].imshow(layout_im))

    plt.show(block=False)

    for x in range(8):
        for y in range(8):
            ground_truth_pob = allo_occupancy_pre_zoomed(
                x=x,
                y=y,
                arr_zoom_padded=arr_zoom_padded,
                pob=pob,
                zoom_level=zoom_level
            )

            images[0].set_data(ground_truth_pob)
            ax_arr[0].figure.canvas.draw()

            time.sleep(1)


def test_gaussian_occupancy(pob=5, res=10):
    import matplotlib.pyplot as plt

    map_arr = layout_to_array(map_layouts[0])

    fig, ax_arr = plt.subplots(2)

    layout_im, layout_arr = layout_to_image(map_layouts[0], res=8)

    images = []

    images.append(ax_arr[0].imshow(np.random.random((res, res))))
    images.append(ax_arr[1].imshow(layout_im))

    plt.show(block=False)

    for x in range(8):
        for y in range(8):
            gauss = allo_occupancy_gaussian(x, y, map_arr, pob, res, std=.5)

            images[0].set_data(gauss)
            ax_arr[0].figure.canvas.draw()

            time.sleep(1)


def test_allo_occ_from_sensors(folder='datasets/hd_50sens_180fov_10map/',
                               pob=5,
                               zoom_level=4,
                               sigma=0.1,
                               n_samples=100,  # stop after displaying this many samples
                               generate_distances=False,  # generate distance measurements
                               ):
    data = np.load(folder + 'data.npy')
    maps = np.load(folder + 'maps.npy')
    labels = pickle.load(open(folder + 'labels.pkl', 'rb'))
    params = json.load(open(folder + 'params.json', 'r'))

    fov = params['fov']
    max_sensor_dist = params['max_sensor_dist']

    # Construct a list of the indices from 'data' that contain distance sensor information
    indices = []
    for i, label in enumerate(labels):
        if 'distsensor' in label:
            indices.append(i)

    # distance sensor measurements for each data point
    # data is saved in a normalized state, so need to multiply by max_sensor_dist
    sensor_data = data[:, indices] * max_sensor_dist

    # index for the map associated to the data
    map_ids = data[:, -1]

    # x,y, and theta for each data point
    xyth = data[:, [0, 1, 2]]

    tile_pos = data[:, [3, 4]]

    # Set up image plots
    fig, ax_arr = plt.subplots(2)

    images = []

    images.append(ax_arr[0].imshow(np.random.random((pob * zoom_level, pob * zoom_level))))
    images.append(ax_arr[1].imshow(maps[0, :, :]))  # Fill with the first map as a default

    plt.show(block=False)

    for index in range(min(len(sensor_data), n_samples)):

        # Generate sensor readings using a custom algorithm rather than use the dataset
        if generate_distances:
            sensor_data[index] = generate_sensor_readings(maps[int(map_ids[index]), :, :],
                                                          zoom_level=4,
                                                          n_sensors=int(params['n_sensors']),
                                                          fov_rad=fov * np.pi / 180.,
                                                          x=xyth[index, 0],
                                                          y=xyth[index, 1],
                                                          th=xyth[index, 2],
                                                          max_sensor_dist=max_sensor_dist,
                                                          )

        occ_arr = allo_occupancy_from_sensors(
            sensor_dists=sensor_data[index],
            pob=pob,
            head_direction=xyth[index, 2],
            fov=fov,
            zoom_level=zoom_level,
            sigma=sigma,
        )

        # Mark the location of the agent with a different colour
        labelled_map = maps[int(map_ids[index]), :, :].copy()
        labelled_map[int(tile_pos[index, 0]), int(tile_pos[index, 1])] = .5

        images[0].set_data(occ_arr)
        # images[1].set_data(maps[int(map_ids[index]), :, :])
        images[1].set_data(labelled_map)

        ax_arr[0].figure.canvas.draw()
        ax_arr[1].figure.canvas.draw()

        time.sleep(1)


# TODO: make a version that re-writes sensor measurement data in old datasets for comparison
def test_sensor_reading_generation():
    # TODO: finish this function
    generate_sensor_readings(map_array,
                             zoom_level=4,
                             n_sensors=30,
                             fov_rad=np.pi,
                             x=0,
                             y=0,
                             th=0,
                             max_sensor_dist=10,
                             )


if __name__ == "__main__":
    # simple tests

    from map_layout import map_layouts

    map_arr = layout_to_array(map_layouts[0])

    test = allo_occupancy(x=2.5, y=2.5, map_arr=map_arr, pob=5, zoom_level=4)

    print(test)

    # test_map_scan()
    # test_gaussian_occupancy()
    test_allo_occ_from_sensors()  # TODO: create a version that runs from a realistic trajectory rather than random points
    # test_allo_occ_from_sensors(generate_distances=True) # TODO: create a version that runs from a realistic trajectory rather than random points
