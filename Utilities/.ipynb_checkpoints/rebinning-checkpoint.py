import numpy as np
from collections import defaultdict
import copy
from scipy import signal
import math


def draw_slice(start_angle, end_angle, image, center, value):
    """
    Given an image, and a center, it draws a pie-like slice
    between angles 'start_angle' and 'end_angle'. The pixels of
    in slice will have the value 'value'.
    """
    start_angle = np.deg2rad(start_angle)
    end_angle = np.deg2rad(end_angle)

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            # calculate the angle of the pixel
            angle = np.arctan2(y - center[1], x - center[0])
            if angle < 0:
                angle += 2 * np.pi

            # this angle check is for the special case were
            # 360 is crossed
            if start_angle > end_angle:
                if angle >= start_angle or angle < end_angle:
                    image[x, y] = value
            else:
                if angle >= start_angle and angle < end_angle:
                    image[x, y] = value

    return image


def draw_wheels(radius, sub_divisions):
    """
    Given a radius and a number of sub-divisions it draws a all the slices
    of the wheel. The main-slices are 10 degrees wide and the sub-divisions
    determine the number of slices in each of the main-slices.
    """
    img = np.zeros((2 * radius, 2 * radius), dtype=np.int16)
    center = (radius - 0.5, radius - 0.5)

    start_angles = station_angles(sub_divisions)
    num_slices = len(start_angles) - 1

    # make the value unique for each slice
    for i in range(num_slices):
        value = i + 1
        img = draw_slice(start_angles[i], start_angles[i + 1], img, center, value)
    return img


def station_angles(sub_divisions):
    """
    The main 'true' slices are 10 degrees wide, divisions determines the sub-divisions of these 10 degree slices (e.g. division= 2 means 5 degree slices, division=3 means 3.33 degree slices)
    """
    start_angle = 45
    end_angle = 360 + 45 + 1

    start_angles = np.array([])
    for i in range(start_angle, end_angle, 10):
        part_angles = np.linspace(i, i + 10, sub_divisions, endpoint=False)
        start_angles = np.concatenate((start_angles, part_angles))

    # Remove any angles that are greater than 360+45 and convert to 0-360 range
    start_angles = start_angles[start_angles <= 360 + 45] % 360
    return start_angles


def make_disk(img, radius, value):
    center = (img.shape[0] // 2 - 0.5, img.shape[1] // 2 - 0.5)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            if distance < radius:
                img[x, y] = value
    return img


def station3_boundaries():
    """
    Returns the inner radius (hole) and the outer radius of the station 3.
    Determined with the method of 'a occhio'
    """
    return 21, 44


def make_rebin_regions(
    mean_matrix, inner_radius, outer_radius, wheel_division, radial_division
):
    empty_img = np.zeros_like(mean_matrix, dtype=np.int16)
    hole = make_disk(empty_img, inner_radius, 1)
    hole = np.abs(hole - 1)
    hole *= make_disk(empty_img, outer_radius, 1)

    radii = np.linspace(inner_radius, outer_radius, radial_division + 1)
    radii = np.floor(radii * 2) / 2

    slices = draw_wheels(mean_matrix.shape[0] // 2, wheel_division)
    conc_disks = make_concentric_disks(
        mean_matrix.shape[0] // 2,
        radii,
    )
    regions = slices * conc_disks * hole

    # here define good area mask
    # inner_disk fills everything just barely out of outer_radius
    inner_disk = make_disk(np.zeros((100, 100)), outer_radius - 1, 1)
    # using mean_matrix is for borders close to outer_radius where
    # the disk-making-function and the data disagree in shape
    good_area_mask = np.where(mean_matrix + inner_disk != 0, 1, 0)
    
    disk = make_disk(np.zeros((100, 100)), inner_radius - 2, 1)
    good_area_mask_2 = np.where(mean_matrix + (1-disk) != 0, 1, 0)
    regions *= good_area_mask*good_area_mask_2

    i, j = np.indices((100, 100))
    distances = np.floor(np.sqrt((i - 49.5)**2 + (j - 49.5)**2)).astype(int)
    regions[distances == 21] = 0

    return regions


def station_radii(divisions):
    # basically use station boundaries and make a list of radii
    # from boundaries[0] to boundaries[1] so that the total number of
    # divisions is respected
    inner_radius = station3_boundaries()[0]
    outer_radius = station3_boundaries()[1]

    radii = np.linspace(inner_radius, outer_radius, divisions + 1)
    # floor to the nearest 0.5
    radii = np.floor(radii * 2) / 2
    return radii


def make_concentric_disks(radius, radii):
    """
    Given a radius and a number of divisions, it draws a disk with concentric
    circles with the number of divisions.
    """
    img = np.zeros((2 * radius, 2 * radius), dtype=np.int16)
    radii = np.sort(radii)[::-1]

    for i, r in enumerate(radii):
        img = make_disk(img, r, i + 257)
    return img


def assign_colors(region_map, num_colors=4):
    neighbors = _get_neighbors(region_map)
    color_assignment = _greedy_coloring(neighbors, num_colors)

    if None in color_assignment.values():
        return assign_colors(region_map, num_colors + 1)

    colored_map = np.vectorize(color_assignment.get)(region_map)
    return colored_map.astype(float) / np.max(colored_map)


def rebin(rebinning, image):
    """
    Example:
    arr1 = np.array([
        [0, 1, 2],
        [1, 2, 3],
        [3, 3, 0]
    ])

    arr2 = np.array([
        [1.3, 2., 4.],
        [2., 6., 8.],
        [7., 6., 3.0]
    ])

    arr3 = [
        [1.3 2.  5. ]
        [2.  5.  7. ]
        [7.  7.  3. ]
    ]
    """
    flat_binning = rebinning.flatten()
    flat_image = image.flatten()

    # Get unique values and inverse indices from flat_binning, excluding 0
    non_zero_indices = flat_binning != 0
    _, inverse_indices = np.unique(flat_binning[non_zero_indices], return_inverse=True)

    # Replace np.nan with 0 in flat_image
    flat_image = np.where(np.isnan(flat_image), 0, flat_image)

    # Sum the values in image corresponding to each unique value in binning
    sum_vals = np.bincount(inverse_indices, weights=flat_image[non_zero_indices])

    # Count the occurrences of each unique value in binning
    count_vals = np.bincount(inverse_indices)

    # Calculate the mean for each unique value
    mean_vals = sum_vals / count_vals

    arr3 = np.copy(image)

    # Fill in the calculated mean values where arr1 is not zero
    arr3[rebinning != 0] = mean_vals[inverse_indices]

    return arr3


def in_out_sel(img, in_=True):
    img_ = img.copy()
    i, j = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')
    dist = np.floor(np.sqrt((i - 49.5)**2 + (j - 49.5)**2)).astype(int)
    
    inside_circle_mask = (dist <= 21)
    outside_circle_mask = (dist >= 21)

    if in_:
        img_[outside_circle_mask] = 0
    else:
        img_[inside_circle_mask] = 0

    return img_
    
def normalize(imgs, entry, lumis, in_=True):
    imgs_ = [in_out_sel(img * (300 / lumi), in_) for img, lumi in zip(imgs, lumis)]
    entry_ = entry*(300/lumis)
    return imgs_, entry_

def normalize_2(row, regions):
    
    new_imgs = rebin(regions, row["img"])  # Usa la colonna 'img' per applicare 'rebin'
    in_img = in_out_sel(new_imgs, True)
    out_img = in_out_sel(new_imgs, False)
    return in_img, out_img
    

def _get_neighbors(region_map):
    rows, cols = region_map.shape
    neighbors = defaultdict(set)

    for r in range(rows):
        for c in range(cols):
            region = region_map[r, c]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and region_map[nr, nc] != region:
                    neighbors[region].add(region_map[nr, nc])
                    neighbors[region_map[nr, nc]].add(region)

    return neighbors


def _greedy_coloring(neighbors, num_colors):
    color_assignment = {}
    for region, region_neighbors in neighbors.items():
        used_colors = {
            color_assignment[neighbor]
            for neighbor in region_neighbors
            if neighbor in color_assignment
        }
        available_colors = set(range(num_colors)) - used_colors
        color_assignment[region] = min(available_colors) if available_colors else None
    return color_assignment


def XY_to_Polar(img):  # No longer used
    """
    Function to change the coordinates of a matrix from (X, Y) to (r, θ).
    """
    imgpolar = np.zeros((34, round(8 * (2 * np.pi))), dtype=np.float32)
    center = (49.5, 49.5)
    for i in range(100):
        for j in range(100):
            if img[i, j] != 0:
                dist = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2) - 11
                theta = np.arctan2((j - center[1]), (i - center[0]))
                theta = theta + np.pi
                try:
                    if round(theta * 8) == round(8 * (2 * np.pi)):
                        imgpolar[round(dist), 0] += img[i, j]
                    else:
                        imgpolar[round(dist), round(theta * 8)] += img[i, j]
                except:
                    pass
    imgpolar[33, :] = 0
    return imgpolar


def rebin_rect(rect, nrows):  # No longer used
    # nrows is the number of rows to be summed together

    # if the rectangle cannot be equally divided into groups of nrows,
    # the last group will have n_last rows

    n_groups = rect.shape[0] // nrows
    n_last = rect.shape[0] % nrows

    main_kernel = np.ones((nrows, 1))
    final_kernel = np.ones((n_last, 1))

    # main_rects[i] are the groups of nrows
    if n_last != 0:
        rebinned_rect = np.zeros((n_groups + 1, rect.shape[1]))
        main_rects = rect[:-n_last].reshape(n_groups, nrows, rect.shape[1])
    else:
        rebinned_rect = np.zeros((n_groups, rect.shape[1]))
        main_rects = rect.reshape(n_groups, nrows, rect.shape[1])

    # summing for the well-divided groups
    for i in range(0, n_groups):
        rebinned_rect[i] = signal.convolve2d(main_rects[i], main_kernel, mode="valid")

    # summing for last group, if present
    if n_last != 0:
        rebinned_rect[-1] = signal.convolve2d(
            rect[-n_last:], final_kernel, mode="valid"
        )

    return rebinned_rect


def rebin_whole(rect, upper_row, nrows_upper, nrows_lower):  # No longer used
    upper_rect = rect[upper_row:,]
    lower_rect = rect[:upper_row,]
    rebinned_upper = rebin_rect(upper_rect, nrows_upper)
    rebinned_lower = rebin_rect(lower_rect, nrows_lower)

    combined_rect = np.vstack((rebinned_lower, rebinned_upper))
    return combined_rect
