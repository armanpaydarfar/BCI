import numpy as np
import mne
import matplotlib.pyplot as plt
import os

# Create a dictionary mapping each channel to its 3D coordinate
def create_channel_coordinate_dict(channel_names, coordinates):
    """
    Create a dictionary mapping channel names to 3D Cartesian coordinates.

    Parameters:
        channel_names (list): List of channel names.
        cartesian_coords_3d (list of tuples): List of (x, y, z) Cartesian coordinates.

    Returns:
        dict: A dictionary mapping channel names to their coordinates.
    """
    return {name: coord for name, coord in zip(channel_names, coordinates)}


def save_as_sfp(filename, chan_info, fiducials):
    """
    Save channel names, 3D coordinates, and fiducials to an .sfp file.

    Parameters:
        filename (str): Name of the .sfp file to save.
        channel_names (list): List of channel names.
        cartesian_coords_3d (list): List of (x, y, z) coordinates for channels.
        fiducials (dict): Dictionary containing fiducial names and (x, y, z) coordinates.
    """
    with open(filename, 'w') as file:
        # Write channel positions
        for chan_name, chan_loc in chan_info.items():
            x, y, z = chan_loc
            file.write(f"{chan_name}\t{x:.4f}\t{y:.4f}\t{z:.4f}\n")
        # Write fiducials
        for fid_name, fid_coord in fiducials.items():
            x, y, z = fid_coord
            file.write(f"{fid_name}\t{x:.4f}\t{y:.4f}\t{z:.4f}\n")
    print(f"Montage with fiducials saved to {filename}")


def load_and_visualize_montage(sfp_filename):
    """
    Load a montage from an .sfp file and visualize it.

    Parameters:
        sfp_filename (str): Path to the .sfp file.
    """
    # Load the montage
    montage = mne.channels.read_custom_montage(sfp_filename)
    print("Loaded montage channel names:", montage.ch_names)

    # Visualize the montage
    montage.plot(show_names=True)



def mirror_y_axis(coords):
    """
    Mirror the given Cartesian coordinates about the y-axis.

    Parameters:
        coords (list of tuples): List of (x, y) coordinates.

    Returns:
        list of tuples: Mirrored coordinates.
    """
    return [(-x, y) for x, y in coords]


def transform_radii_to_absolute(r_theta_coords, circle_radius):
    """
    Transform radii and angles (r, theta) into absolute Cartesian coordinates,
    assuming the radii are proportions of the given circle radius.

    Parameters:
        r_theta_coords (list of tuples): List of (theta, r) values where r is a proportion.
        circle_radius (float): The actual radius of the circle.

    Returns:
        list of tuples: Transformed Cartesian coordinates (x, y).
    """
    return [
        (r * circle_radius * np.cos(np.radians(theta)), r * circle_radius * np.sin(np.radians(theta)))
        for theta, r in r_theta_coords
    ]



# Function to rotate positions
def rotate_positions(positions, angle_degrees):
    """
    Rotate a list of 2D Cartesian positions by a given angle in degrees.

    Parameters:
        positions (list of tuple): List of (x, y) positions to rotate.
        angle_degrees (float): Angle by which to rotate the positions, in degrees.

    Returns:
        list of tuple: Rotated positions.
    """
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])

    # Apply the rotation
    rotated_positions = [tuple(np.dot(rotation_matrix, np.array([x, y]))) for x, y in positions]
    return rotated_positions

# R and Theta coordinates (Theta in degrees)
r_theta_coords = [
    (-17.926, 0.514988888888889),
    (0, 0.506688888888889),
    (17.926, 0.514988888888889),
    (-53.913, 0.528083333333333),
    (-39.947, 0.344594444444444),
    (0, 0.253377777777778),
    (39.897, 0.344500000000000),
    (53.867, 0.528066666666667),
    (-69.332, 0.408233333333333),
    (-44.925, 0.181183333333333),
    (44.925, 0.181183333333333),
    (69.332, 0.408233333333333),
    (-100.420, 0.747350000000000),
    (-90, 0.533183333333333),
    (-90, 0.266688888888889),
    (0, 0),
    (90, 0.266666666666667),
    (90, 0.533183333333333),
    (100.420, 0.747350000000000),
    (-110.668, 0.408233333333333),
    (-135.075, 0.181183333333333),
    (135.075, 0.181183333333333),
    (110.668, 0.408233333333333),
    (-126.087, 0.528083333333333),
    (-140.053, 0.344594444444444),
    (180, 0.253377777777778),
    (140.103, 0.344500000000000),
    (126.133, 0.528066666666667),
    (180, 0.379944444444445),
    (-162.074, 0.514988888888889),
    (180, 0.506688888888889),
    (162.074, 0.514988888888889),
]

# Channel names
channel_names = [
    'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz',
    'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7',
    'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2'
]

# Fiducials
fiducials = {
    "nasion": [0.0, 0.1, 0.0],
    "lpa": [-0.1, 0.0, 0.0],
    "rpa": [0.1, 0.0, 0.0],
}

# Convert R and Theta to Cartesian coordinates (Theta in degrees -> radians)
cartesian_coords = transform_radii_to_absolute(r_theta_coords,0.14)

# Rotate Cartesian coordinates by -90 degrees
angle_to_rotate = 90
rotated_cartesian_coords = rotate_positions(cartesian_coords, angle_to_rotate)
mirrored_cartesian_coords = mirror_y_axis(rotated_cartesian_coords)
# Create a dictionary for channel positions (only x and y)
channel_positions = {name: pos for name, pos in zip(channel_names, mirrored_cartesian_coords)}
# Create dummy data for the channels
data = np.zeros(len(channel_names))  # Example data for the topo plot

# Plot the topomap
fig, ax = plt.subplots(figsize=(8, 8))
mne.viz.plot_topomap(
    data,
    np.array(list(channel_positions.values())) ,  # Convert to meters
    axes=ax,
    names=channel_names,
    contours=0  # Disable contour lines for clarity
)
ax.set_title(f'Topographic Map (Rotated {angle_to_rotate}°)', fontsize=16)
plt.show()

cartesian_coords_3d = [(x, y, 0) for x, y in mirrored_cartesian_coords]

coord_dict = create_channel_coordinate_dict(channel_names, cartesian_coords_3d)
print(coord_dict)
# Create the montage
# Fiducial points (in meters)
nasion = [0.0, 0.1, 0.0]
lpa = [-0.1, 0.0, 0.0]
rpa = [0.1, 0.0, 0.0]
montage = mne.channels.make_dig_montage(
    ch_pos=coord_dict,
    nasion=nasion,
    lpa=lpa,
    rpa=rpa,
    coord_frame='head'
)
montage.plot(kind='topomap', show_names=True)
plt.show()

# Save the montage as a .fif file
montage_path = "CA-209-dig.fif"
montage.save(montage_path, overwrite=True)
print(f"Montage saved to {montage_path}")

loaded_montage = mne.channels.read_dig_fif("CA-209-dig.fif")
print("Loaded montage channel names:", loaded_montage.ch_names)

# Visualize the montage
loaded_montage.plot(kind='topomap', show_names=True)
plt.show()