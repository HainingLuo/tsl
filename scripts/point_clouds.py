import struct
import open3d
import numpy as np
from math import isnan
from ctypes import * # convert float to uint32

import rospy
import ros_numpy
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2

# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]

# Bit operations
BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8
convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)
datatype = {1:1, 2:1, 3:2, 4:2, 5:4, 6:4, 7:4, 8:8}

def o3dpc_to_pc2(o3dpc, frame_id="odom"):
    """ Convert the datatype of point cloud from Open3D to ROS PointCloud2 (XYZRGB only) """
    # Set "header"
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    # Set "fields" and "cloud_data"
    points=np.asarray(o3dpc.points)
    if not o3dpc.colors: # XYZ only
        fields=FIELDS_XYZ
        cloud_data=points
    else: # XYZ + RGB
        fields=FIELDS_XYZRGB
        # -- Change rgb color from "three float" to "one 24-byte int"
        # 0x00FFFFFF is white, 0x00000000 is black.
        colors = np.floor(np.asarray(o3dpc.colors)*255) # nx3 matrix
        colors = colors[:,0] * BIT_MOVE_16 +colors[:,1] * BIT_MOVE_8 + colors[:,2]  
        cloud_data=np.c_[points, colors]
    
    # create ros_cloud
    return pc2.create_cloud(header, fields, cloud_data)

def pc2_to_o3dpc(pc_msg, remove_nans=False):
    """ covert ros point cloud to open3d point cloud
    Args: 
        pc_msg (sensor.msg.PointCloud2): ros point cloud message
        remove_nans (bool): if true, ignore the NaN points
    Returns: 
        o3dpc (open3d.geometry.PointCloud): open3d point cloud
    """
    field_names = [field.name for field in pc_msg.fields]
    is_rgb = 'rgb' in field_names
    cloud_array = ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg).ravel()
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]
    if is_rgb:
        cloud_npy = np.zeros(cloud_array.shape + (4,), dtype=np.float)
    else: 
        cloud_npy = np.zeros(cloud_array.shape + (3,), dtype=np.float)
    
    cloud_npy[...,0] = cloud_array['x']
    cloud_npy[...,1] = cloud_array['y']
    cloud_npy[...,2] = cloud_array['z']
    o3dpc = open3d.geometry.PointCloud()

    if len(np.shape(cloud_npy)) == 3:
        cloud_npy = np.reshape(cloud_npy[:, :, :3], [-1, 3], 'F')
    o3dpc.points = open3d.utility.Vector3dVector(cloud_npy[:, :3])

    if is_rgb:
        rgb_npy = cloud_array['rgb']
        rgb_npy.dtype = np.uint32
        r = np.asarray((rgb_npy >> 16) & 255, dtype=np.uint8)
        g = np.asarray((rgb_npy >> 8) & 255, dtype=np.uint8)
        b = np.asarray(rgb_npy & 255, dtype=np.uint8)
        rgb_npy = np.asarray([r, g, b])
        rgb_npy = rgb_npy.astype(np.float)/255
        rgb_npy = np.swapaxes(rgb_npy, 0, 1)
        o3dpc.colors = open3d.utility.Vector3dVector(rgb_npy)
    return o3dpc

def pointcloud2_to_array(cloud_msg, squeeze=True):
    ''' Converts a rospy PointCloud2 message to a numpy recordarray 
    
    Reshapes the returned array to have shape (height, width), even if the height is 1.
    The reason for using np.frombuffer rather than struct.unpack is speed... especially
    for large point clouds, this will be <much> faster.
    '''
    # construct a numpy record type equivalent to the point type of this cloud
    dtype_list = fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)

    # parse the cloud into an array
    cloud_arr = np.frombuffer(cloud_msg.data, dtype_list)

    # remove the dummy fields that were added
    cloud_arr = cloud_arr[
        [fname for fname, _type in dtype_list if not (fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]
    
    if squeeze and cloud_msg.height == 1:
        return np.reshape(cloud_arr, (cloud_msg.width,))
    else:
        return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))

def read_point(point_2d, pc):
    """ get 3d point position from sensor_msgs point cloud """  
    arrayPosition = point_2d[0]*pc.row_step + point_2d[1]*pc.point_step # point_2d: y,x
    pos_x = arrayPosition + pc.fields[0].offset # X has an offset of 0
    len_x = datatype[pc.fields[0].datatype]
    pos_y = arrayPosition + pc.fields[1].offset # Y has an offset of 4
    len_y = datatype[pc.fields[1].datatype]
    pos_z = arrayPosition + pc.fields[2].offset # Z has an offset of 8
    len_z = datatype[pc.fields[2].datatype]

    try:
        x = struct.unpack('f', pc.data[pos_x: pos_x+len_x])[0] # read 4 bytes as a float number
        y = struct.unpack('f', pc.data[pos_y: pos_y+len_y])[0]
        z = struct.unpack('f', pc.data[pos_z: pos_z+len_z])[0]
        return [x,y,z]
    except:
        return None

def depth_pixel_to_metric_coordinate(point_2d, depth_image, camera_intrinsics):
    """ 
    [get 3D coordinate from depth image]
    Input: point 2d (list of doubles) (y and x value of the image coordinate), depth (double), 
        camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
        |fx, 0,  ppx|
        |0,  fy, ppy|
        |0 , 0,  1  |
    Output: point 3d (x,y,z in meters) list of doubles
    """
    [height, width] = depth_image.shape
    depth = depth_image[point_2d[0],point_2d[1]]/1000
    X = (point_2d[1] - camera_intrinsics[2])/camera_intrinsics[0]*depth
    Y = (point_2d[0] - camera_intrinsics[5])/camera_intrinsics[4]*depth
    return [X, Y, depth]

def depth_pixels_to_metric_coordinates(points_2d, depth_image, camera_intrinsics):
    """ 
    [get 3D coordinate from depth image]
    Input: point 2d (list of doubles) (y and x value of the image coordinate), depth (double), 
        camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
        |fx, 0,  ppx|
        |0,  fy, ppy|
        |0 , 0,  1  |
    Output: point 3d (x,y,z in meters) list of doubles
    """
    [height, width] = depth_image.shape
    depth = depth_image[points_2d[:,0],points_2d[:,1]]/1000
    X = (points_2d[:,1] - camera_intrinsics[2])/camera_intrinsics[0]*depth
    Y = (points_2d[:,0] - camera_intrinsics[5])/camera_intrinsics[4]*depth
    return np.transpose([X, Y, depth])

def depth_pixels_to_metric_coordinate(points_2d, depth_image, camera_intrinsics, mode='mean'):
    """ 
    [get 3D coordinate from depth image]
    Input: point 2d (list of doubles) (y and x value of the image coordinate), depth (double), 
        camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
        |fx, 0,  ppx|
        |0,  fy, ppy|
        |0 , 0,  1  |
    Output: point 3d (x,y,z in meters) list of doubles
    """
    # [height, width] = depth_image.shape
    point_2d = np.mean(points_2d, axis=0)
    depths = depth_image[points_2d[:,0],points_2d[:,1]].flatten()/1000
    valid_ids = np.argwhere((depths!=np.nan) & (depths!=0))
    # valid_ids = ((not np.isnan(depths))& (depths!=0)).nonzero()[0]
    depths = depths[valid_ids].squeeze()
    if mode=='mean':
        depth = np.mean(depths)
    elif mode=='middle':
        depth = depths[depths.argsort()][depths.size//2]
    else:
        print('Unknown Mode! Returning mean instead!')
        depth = depths.mean(axis=0)

    X = (point_2d[1] - camera_intrinsics[2])/camera_intrinsics[0]*depth
    Y = (point_2d[0] - camera_intrinsics[5])/camera_intrinsics[4]*depth
    return np.transpose([X, Y, depth])

def metric_coordinate_to_pixel_coordinate(point_3d, camera_intrinsics):
    '''
    Input: 
    - point 3d (x,y,z in meters) list of doubles
    Output:
    - point 2d: y,x
    '''
    [X, Y, depth] = point_3d
    x = X/depth*camera_intrinsics[0] + camera_intrinsics[2]
    y = Y/depth*camera_intrinsics[4] + camera_intrinsics[5]
    return np.array([int(y),int(x)])

def check_validity(point_2d, pc):
    return ((point_2d[0]>=500) and (point_2d[1]<=1300) and (point_2d[1]>=550))

def read_point_from_region(point_2d, pc, region, mode='mean', camera_intrinsics=None):
    """ 
    [get a middle 3d point position from a region within point cloud]
    Input: point 2d (row, col), point cloud message, rectangular edge length
    Output: point 3d (x,y,z)
    """
    # assert all(point_2d), '2D target should not be on the edges!'
    point_2d = np.array(point_2d, dtype=np.int32)
    row, col = np.indices((region, region))-region//2
    grid_ids = np.transpose([row.flatten()+point_2d[0], col.flatten()+point_2d[1]])
    if camera_intrinsics is None:
        points_3d = []
        for id in grid_ids:
            if camera_intrinsics==None:
                point_3d = read_point(id, pc)
                if not isnan(point_3d[0]): points_3d.append(point_3d)
        # sort the positions according to the y values
        if len(points_3d)>0:
            points_3d = np.array(points_3d)
            if mode=='mean':
                return points_3d.mean(axis=0)
            elif mode=='middle':
                return points_3d[points_3d[:,1].argsort()][points_3d.shape[0]//2]
            else:
                print('Unknown Mode! Returning mean instead!')
                return points_3d.mean(axis=0)
    else:
        return depth_pixels_to_metric_coordinate(grid_ids, pc, camera_intrinsics, mode=mode)

def read_points_from_region(point_2d_list, pc, region, mode='mean', camera_intrinsics=None):
    """ get 3d point positions from a region within point cloud """
    points_3d = []
    for point_2d in point_2d_list:
        points_3d.append(read_point_from_region(point_2d, pc, region=region, mode=mode, camera_intrinsics=camera_intrinsics))
    return np.array(points_3d)
        
def depth_to_pointcloud2(depth_image, camera_intrinsics, stamp=None, frame_id=None, image=None):
    """
    convert depth map into point cloud
    """
    if image is None:
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1)]
    else:
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgb', 12, PointField.UINT32, 1)]

    header = Header()
    header.frame_id = frame_id
    header.stamp = stamp

    rows, cols = depth_image.shape
    points_2d = np.indices(depth_image.shape) # rows and cols
    depth = depth_image[points_2d[0],points_2d[1]]/1000
    X = (points_2d[1] - camera_intrinsics[2])/camera_intrinsics[0]*depth
    Y = (points_2d[0] - camera_intrinsics[5])/camera_intrinsics[4]*depth
    if image is None:
        # points = np.moveaxis([X,Y,depth], -1, 0).reshape((3, -1)).T
        points = []
        for i in range(rows):
            for j in range(cols):
                points.append([X[i,j], Y[i,j], depth[i,j]])
        # points = np.array([X,Y,depth]).reshape((-1, 3))
    else:
        img_rgba = np.concatenate((image.astype(np.uint8), np.full((*image.shape[:-1], 1), 0, dtype=np.uint8)), axis=-1)
        rgb = np.squeeze(img_rgba.view(np.uint32))

        points = []
        for i in range(rows):
            for j in range(cols):
                # rgb = struct.unpack('I', struct.pack('BBBB', image[i,j,0], image[i,j,1], image[i,j,2], 255))[0]
                points.append([X[i,j], Y[i,j], depth[i,j], rgb[i,j]])
                
        # points = np.array([X, Y, depth]).astype(object)
        # points = np.concatenate([points, rgb], axis=-1).reshape((-1, 4))
    return point_cloud2.create_cloud(header, fields, points)
    # return ros_numpy.point_cloud2.array_to_pointcloud2(points, stamp=stamp, frame_id=frame_id)

def xy_to_yx(points):
    return np.flip(points, axis=-1)

def euclidian_distance(point1, point2):
    return sum([(point1[x] - point2[x]) ** 2 for x in range(len(point1))]) ** 0.5

def is_sorted(a):
    ''' check if a list is in ascending order '''
    return np.all(a[:-1] <= a[1:])