import numpy as np
from track_dlo_utils import extract_connected_skeleton
from order_gaussian_centres import order_gaussian_centres
from sklearn.mixture import GaussianMixture
from math import isnan

from point_clouds import read_points_from_region


def estimate_initial_states_si(cloud_pts_2d, cloud_pts_3d, depth, camera_intrinsics, num_ctrl_pts): 
    # print the dimensions of the input
    print(f"cloud_pts_2d: {cloud_pts_2d.shape}")
    print(f"cloud_pts_3d: {cloud_pts_3d.shape}")

    import matplotlib.pyplot as plt
    mask = cloud_pts_2d.squeeze()
    # plot the corners of cloud points in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cloud_pts_3d[0, 0, 0], cloud_pts_3d[0, 0, 1], cloud_pts_3d[0, 0, 2], c='r', s=10)
    ax.scatter(cloud_pts_3d[0, -1, 0], cloud_pts_3d[0, -1, 1], cloud_pts_3d[0, -1, 2], c='b', s=10)
    ax.scatter(cloud_pts_3d[-1, 0, 0], cloud_pts_3d[-1, 0, 1], cloud_pts_3d[-1, 0, 2], c='g', s=10)
    ax.scatter(cloud_pts_3d[-1, -1, 0], cloud_pts_3d[-1, -1, 1], cloud_pts_3d[-1, -1, 2], c='y', s=10)
    plt.show()
    
    ############################# From TRACKDLO #############################
    extracted_chains = extract_connected_skeleton(False, mask>0, img_scale=1, \
                                                    seg_length=3, max_curvature=25)
    all_pixel_coords = []
    for chain in extracted_chains:
        all_pixel_coords += chain

    # plot the pixel coordinates
    all_pixel_coords = np.array(all_pixel_coords)
    plt.scatter(all_pixel_coords[:, 0], all_pixel_coords[:, 1], s=1)
    # # connect the points
    # for i in range(len(all_pixel_coords)-1):
    #     plt.plot(all_pixel_coords[i:i+2, 0], all_pixel_coords[i:i+2, 1], 'r')
    plt.show()
        
    # flip the order of the coordinates to x,y
    # all_pixel_coords = np.flip(np.array(all_pixel_coords), axis=1) 
    all_pixel_coords = np.array(all_pixel_coords)
    # convert 2D pixel coordinates to 1D indices
    # all_pixel_coords = np.ravel_multi_index(all_pixel_coords.T, mask.shape)
    # extract the 3D points
    extracted_chains_3d = get_cloud_pts(all_pixel_coords, depth, camera_intrinsics)
    # extracted_chains_3d = cloud_pts_3d[all_pixel_coords[:, 1], all_pixel_coords[:, 0]]
    # remove the zero points
    # extracted_chains_3d = extracted_chains_3d[((extracted_chains_3d[:, 0] != 0) | \
    #                                             (extracted_chains_3d[:, 1] != 0) | \
    #                                             (extracted_chains_3d[:, 2] != 0))]
    
    # plot the 3D points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(extracted_chains_3d[:, 0], extracted_chains_3d[:, 1], extracted_chains_3d[:, 2], s=1)
    plt.show()

    from scipy import interpolate
    tck, u = interpolate.splprep(extracted_chains_3d.T, s=0.0005)
    # 1st fit, less points
    u_fine = np.linspace(0, 1, 300) # <-- num fit points
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
    spline_pts = np.vstack((x_fine, y_fine, z_fine)).T

    # 2nd fit, higher accuracy
    num_true_pts = int(np.sum(np.sqrt(np.sum(np.square(np.diff(spline_pts, axis=0)), axis=1))) 
                        * 1000)
    u_fine = np.linspace(0, 1, num_true_pts) # <-- num true points
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
    spline_pts = np.vstack((x_fine, y_fine, z_fine)).T

    initial_states = spline_pts[np.linspace(0, num_true_pts-1, num_ctrl_pts).astype(int)]
    ##########################################################################
    # reverse initial states if head is to the right of the end
    if initial_states[0][1] > initial_states[-1][1]:
        initial_states = np.flipud(initial_states)

    # plot the initial states
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(initial_states[:, 0], initial_states[:, 1], initial_states[:, 2], s=10)
    for i in range(len(initial_states)-1):
        ax.plot(initial_states[i:i+2, 0], initial_states[i:i+2, 1], initial_states[i:i+2, 2], 'r')
    plt.show()
    return initial_states


def estimate_initial_states_ga(cloud_pts, num_ctrl_pts):
    # import time
    # # time the function
    # start = time.time()

    # fit the gaussian mixture model
    gm = GaussianMixture(n_components=num_ctrl_pts, random_state=0, max_iter=10000).fit(cloud_pts)

    # # print the time taken
    # print(f"Time taken to fit GMM: {time.time()-start:.2f} seconds")
    # start = time.time()

    initial_states = gm.means_
    # sort the centres
    initial_states = initial_states[initial_states[:, 0].argsort()]
    initial_states = initial_states[order_gaussian_centres(initial_states)]
    # reverse initial states if head is to the right of the end
    if initial_states[0][1] > initial_states[-1][1]:
        initial_states = np.flipud(initial_states)

    # # print the time taken
    # print(f"Time taken to estimate initial states: {time.time()-start:.2f} seconds")
    return initial_states


def get_cloud_pts(pixel_pts, depth, camera_intrinsics):
    cloud_pts = read_points_from_region(pixel_pts, depth, region=3, camera_intrinsics=camera_intrinsics)
    return np.array([cloud_pt for cloud_pt in cloud_pts if not isnan(cloud_pt[0])]) # remove Nones