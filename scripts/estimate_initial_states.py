import numpy as np
from track_dlo_utils import extract_connected_skeleton
from order_gaussian_centres import order_gaussian_centres
from sklearn.mixture import GaussianMixture

# def estimate_initial_states_trackdlo(mask):
#     ############################# From TRACKDLO #############################
#     extracted_chains = extract_connected_skeleton(False, mask>0, img_scale=1, \
#                                                     seg_length=5, max_curvature=25)
#     all_pixel_coords = []
#     for chain in extracted_chains:
#         all_pixel_coords += chain
#     # flip the order of the coordinates to x,y
#     all_pixel_coords = np.flip(np.array(all_pixel_coords), axis=1) 
#     extracted_chains_3d = self.get_cloud_pts(all_pixel_coords, depth)
#     extracted_chains_3d = extracted_chains_3d[((extracted_chains_3d[:, 0] != 0) | \
#                                                 (extracted_chains_3d[:, 1] != 0) | \
#                                                 (extracted_chains_3d[:, 2] != 0))]

#     from scipy import interpolate
#     tck, u = interpolate.splprep(extracted_chains_3d.T, s=0.0005)
#     # 1st fit, less points
#     u_fine = np.linspace(0, 1, 300) # <-- num fit points
#     x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
#     spline_pts = np.vstack((x_fine, y_fine, z_fine)).T

#     # 2nd fit, higher accuracy
#     num_true_pts = int(np.sum(np.sqrt(np.sum(np.square(np.diff(spline_pts, axis=0)), axis=1))) 
#                         * 1000)
#     u_fine = np.linspace(0, 1, num_true_pts) # <-- num true points
#     x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
#     spline_pts = np.vstack((x_fine, y_fine, z_fine)).T

#     initial_states = spline_pts[np.linspace(0, num_true_pts-1, self.num_ctrl_pts).astype(int)]
#     ##########################################################################
#     # reverse initial states if head is to the right of the end
#     if initial_states[0][1] > initial_states[-1][1]:
#         initial_states = np.flipud(initial_states)
#     return initial_states


def estimate_initial_states(cloud_pts, num_ctrl_pts):    
    # fit the gaussian mixture model
    gm = GaussianMixture(n_components=num_ctrl_pts, random_state=0, max_iter=10000).fit(cloud_pts)
    initial_states = gm.means_
    # sort the centres
    initial_states = initial_states[initial_states[:, 0].argsort()]
    initial_states = initial_states[order_gaussian_centres(initial_states)]
    # reverse initial states if head is to the right of the end
    if initial_states[0][1] > initial_states[-1][1]:
        initial_states = np.flipud(initial_states)
    return initial_states