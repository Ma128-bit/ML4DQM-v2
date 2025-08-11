import numpy as np

def main(df, me_name, ring_id, num_sectors= 160, name="custom_img"):
    df[name] = df.apply( lambda row: chamber_sel(row["img"], me_name, ring_id), axis=1)
    binary_matrix = (np.mean(df[name], axis=0) != 0)

    if ring_id > 1:
        df[name] = df.apply( lambda row: rebin_image(row[name], binary_matrix, num_sectors), axis=1)

    mean_entries_norm = df["entries"].mean()
    std_entries_norm = df["entries"].std()
    bad_flag = df["entries"].apply(lambda e: 0 if abs(e - mean_entries_norm) < 4 * std_entries_norm else 1)
    return bad_flag
        

def chamber_sel(matrix, me_name, ring_id):
    if me_name.endswith("1"):
        n_rings = 3
        radius = [20, 40]
    else:
        n_rings = 2
        radius = [21]
    if (ring_id > n_rings) | (ring_id < 1):
        raise ValueError(f"Invalid ring_id: {ring_id}. Must be between 1 and {n_rings}")
    if ring_id == 1:
        matrix = mask_region(matrix, range_=(0, radius[0]), center=(49.5, 49.5))
    if ring_id == 2:
        if n_rings == 2:
            matrix = mask_region(matrix, range_=(radius[0], 70), center=(49.5, 49.5))
        else:
            matrix = mask_region(matrix, range_=(radius[0], radius[1]), center=(49.5, 49.5))
    if ring_id == 3:
        matrix = mask_region(matrix, range_=(radius[1], 70), center=(49.5, 49.5))
    return matrix


def mask_region(img, range_, center=(49.5, 49.5)):
    x_center, y_center = center
    d_min, d_max = range_
    ny, nx = img.shape 

    x, y = np.meshgrid(np.arange(nx), np.arange(ny))

    distance = np.sqrt((x - x_center)**2 + (y - y_center)**2).astype(int)
    
    masked_img = np.where(((distance < d_max) & (distance > d_min)), img, 0)

    return masked_img

def rebin_image(image, binary_matrix, num_sectors= 160):    
    angles = np.linspace(0, 2 * np.pi, num_sectors, endpoint=False)
    r, c = image.shape
    center = (49.5, 49.5)
    
    # Create polar coordinate grid
    y, x = np.indices((r, c))
    theta = np.arctan2(y - center[0], x - center[1]) -np.radians(25) + np.pi/num_sectors  # Polar angle
    theta[theta < 0] += 2 * np.pi  # Ensure all angles are positive

    # Sector mapping
    rebinning = np.digitize(theta, angles)
    rebinning = rebinning * binary_matrix

    flat_binning = rebinning.flatten()
    flat_image = image.flatten()

    non_zero_indices = flat_binning != 0
    _, inverse_indices = np.unique(flat_binning[non_zero_indices], return_inverse=True)

    flat_image = np.where(np.isnan(flat_image), 0, flat_image)

    sum_vals = np.bincount(inverse_indices, weights=flat_image[non_zero_indices])

    count_vals = np.bincount(inverse_indices)

    mean_vals = sum_vals / count_vals

    arr3 = np.copy(image)

    arr3[rebinning != 0] = mean_vals[inverse_indices]
    arr3[rebinning == 0] = 0

    return arr3


def rebin_image_to_vector(image, binary_matrix, num_sectors=160):    
    angles = np.linspace(0, 2 * np.pi, num_sectors, endpoint=False)
    r, c = image.shape
    center = (49.5, 49.5)
    
    # Create polar coordinate grid
    y, x = np.indices((r, c))
    theta = np.arctan2(y - center[0], x - center[1]) - np.radians(25) + np.pi / num_sectors  # Polar angle
    theta[theta < 0] += 2 * np.pi  # Ensure all angles are positive

    # Sector mapping
    rebinning = np.digitize(theta, angles)
    rebinning = rebinning * binary_matrix  # Apply binary mask

    flat_binning = rebinning.flatten()
    flat_image = image.flatten()

    non_zero_indices = flat_binning != 0
    bin_indices, inverse_indices = np.unique(flat_binning[non_zero_indices], return_inverse=True)

    flat_image = np.where(np.isnan(flat_image), 0, flat_image)

    sum_vals = np.bincount(inverse_indices, weights=flat_image[non_zero_indices])
    count_vals = np.bincount(inverse_indices)

    mean_vals = np.zeros(num_sectors)
    mean_vals[bin_indices - 1] = sum_vals / count_vals  # Map values to correct indices

    return mean_vals