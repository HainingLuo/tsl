#!/usr/bin/env python3
import io
import cv2
import numpy as np
from os import path
from tqdm import tqdm
# from imutils import resize
from skimage.morphology import medial_axis
from math import isnan
import matplotlib.pyplot as plt

# import tf
import rospy
import rosbag
import rospkg
np.float = np.float64
from ros_numpy import numpify, msgify
# from tf import transformations
from std_msgs.msg import Float64MultiArray, Header
from sensor_msgs.msg import Image, PointCloud2, PointField
from state_estimation_msgs.srv import *

from vision_utils.point_clouds import read_point
from vision_utils.point_clouds import read_points_from_region
from vision_utils.colour_segmentation_hsv import ColourSegmentation
from utils import extract_connected_skeleton, ndarray2MarkerArray
    
""" transforms a numpy array into a sensor_msgs/Image message """
def msgify(img, encoding="bgr8"):
    # height, width = np.size(img)[:2] # H,W,C
    msg = Image()
    msg.header.stamp = rospy.Time.now()
    msg.height = np.shape(img)[0]
    msg.width = np.shape(img)[1]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = int(np.size(img)/np.shape(img)[0])
    msg.data = img.tobytes()
    return msg
    
def numpify_img_msg(msg):
    msg.__class__ = Image
    img = numpify(msg)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # return numpify(msg)

def numpify_depth_msg(msg):
    msg.__class__ = Image
    return numpify(msg)
    # return numpify(msg, encoding='mono8')

class PointRegistrationNode:
    camera_name = 'rs_l515'
    robot_frame = "/yumi_base_link"
    registration_srv_topic = '/registration'
    package_name = 'state_estimation_client'
    num_ctrl_pts = 25
    sample_density = 500 # the number of points in X
    datatype = {1:1, 2:1, 3:2, 4:2, 5:4, 6:4, 7:4, 8:8}
    insertion_frame_ids = [79, 333, 754, 1191]

    def __init__(self):

        self.image_topic = f'/{self.camera_name}/color/image_raw'
        self.depth_topic = f'/{self.camera_name}/aligned_depth_to_color/image_raw'
        self.image_topic_compressed = f'/{self.camera_name}/color/image_raw/compressed'
        self.depth_topic_compressed = f'/{self.camera_name}/aligned_depth_to_color/image_raw/compressed'
        self.info_topic = f'/{self.camera_name}/aligned_depth_to_color/camera_info'
        self.frame = f'{self.camera_name}_color_optical_frame'
        self.tsl_output_topic = '/tsl/result_states'
        self.tsl_input_topic = '/tsl/segmented_pc'
        self.bag_name = 'manip.bag'

        # #################################### Video recording ##################################
        # self.record_video = False
        # self.record_video_resolution = (2880, 540)
        # self.record_video_fn = path.join(path.dirname(path.realpath(__file__)), "video.mp4")
        # self.record_video_fps = 30
        # self.video = cv2.VideoWriter(self.record_video_fn, 0x7634706d, self.record_video_fps, self.record_video_resolution)
        # #################################### Video recording ##################################

        # register publishers, subscribers and services
        self.new_states = False
        self.pts_pub = rospy.Publisher(self.tsl_input_topic, PointCloud2, queue_size=10)
        self.mask_pub = rospy.Publisher('tsl/mask', Image, queue_size=10)
        self.img_pub = rospy.Publisher('tsl/img', Image, queue_size=10)
        rospy.Subscriber(self.tsl_output_topic, PointCloud2, self.cb)
        rospy.sleep(1)

        # load initial data
        self.data_path = path.join(rospkg.RosPack().get_path(self.package_name), 'data')
        self.bag = rosbag.Bag(path.join(self.data_path, self.bag_name))
        self.camera_intrinsics = next(self.bag.read_messages(topics=[self.info_topic]))[1].K
        self.img_msg_iter = self.bag.read_messages(topics=[self.image_topic])
        self.depth_msg_iter = self.bag.read_messages(topics=[self.depth_topic])
        # read the first frame
        n_msgs = self.bag.get_message_count(self.image_topic)
        # for _ in range(n_msgs//3*2):
        #     _, img_msg, _ = next(self.img_msg_iter)
        img_msg = next(self.img_msg_iter).message
        depth_msg = next(self.depth_msg_iter).message
        img = numpify_img_msg(img_msg)
        depth = numpify_depth_msg(depth_msg)
        read_img = lambda : img
        self.segmentation = ColourSegmentation([22, 126, 73] , [57, 255, 214], lambda : read_img, live_adjust=False, morph_op=[2,2,2,3,3,3]) # manip hsv demo
        print(self.segmentation.thresh_l, self.segmentation.thresh_h)
        print('segmentation ready')     
        self.ori_states = None

        # estimate the initial states
        img = self.process_img(img)
        mask = self.segmentation.predict_img(img)
        self.img_pub.publish(msgify(img, 'bgr8'))
        self.mask_pub.publish(msgify(mask, 'mono8'))

        ############################# From TRACKDLO #############################
        extracted_chains = extract_connected_skeleton(False, mask, img_scale=1, seg_length=5, max_curvature=25)
        all_pixel_coords = []
        for chain in extracted_chains:
            all_pixel_coords += chain
        all_pixel_coords = np.flip(np.array(all_pixel_coords), axis=1) # flip the order of the coordinates to x,y
        extracted_chains_3d = self.get_cloud_pts(all_pixel_coords, depth)
        extracted_chains_3d = extracted_chains_3d[((extracted_chains_3d[:, 0] != 0) | (extracted_chains_3d[:, 1] != 0) | (extracted_chains_3d[:, 2] != 0))]

        from scipy import interpolate
        tck, u = interpolate.splprep(extracted_chains_3d.T, s=0.0005)
        # 1st fit, less points
        u_fine = np.linspace(0, 1, 300) # <-- num fit points
        x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
        spline_pts = np.vstack((x_fine, y_fine, z_fine)).T

        # 2nd fit, higher accuracy
        num_true_pts = int(np.sum(np.sqrt(np.sum(np.square(np.diff(spline_pts, axis=0)), axis=1))) * 1000)
        u_fine = np.linspace(0, 1, num_true_pts) # <-- num true points
        x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
        spline_pts = np.vstack((x_fine, y_fine, z_fine)).T

        initial_states = spline_pts[np.linspace(0, num_true_pts-1, self.num_ctrl_pts).astype(int)]
        ##########################################################################

        initial_state_msg = self.pts_to_pointcloud_msg(initial_states)

        # call tsl
        self.new_states = False
        self.pts_pub.publish(initial_state_msg)
        # wait for tsl publishes results
        while not self.new_states and not rospy.is_shutdown():
            rospy.sleep(0.1)
        # states = rospy.wait_for_message(self.tsl_output_topic, PointCloud2)
        # res = self.plot_results(initial_states, states)
        # cv2.imshow('plot', res)

        # start the main loop
        print('start the main loop')
        data_pair = {'img':None, 'depth':None}
        cnt = 0
        try:
            for topic, msg, t in tqdm(self.bag.read_messages(topics=[self.image_topic, self.depth_topic]), total=n_msgs, unit='images'):
                # data_pair
                if topic == self.image_topic:
                    data_pair['img'] = numpify_img_msg(msg)
                if topic == self.depth_topic:
                    data_pair['depth'] = numpify_depth_msg(msg)

                if data_pair['img'] is not None and data_pair['depth'] is not None:
                    # call segmentation
                    img = self.process_img(data_pair['img'])
                    mask = self.segmentation.predict_img(img)
                    self.img_pub.publish(msgify(img, 'bgr8'))
                    self.mask_pub.publish(msgify(mask, 'mono8'))

                    # extract points from masks
                    pixel_pts = np.transpose(np.nonzero(mask)) # consider the target points are white (row,column = y,x)
                    pixel_pts = np.random.permutation(pixel_pts)[:min(self.sample_density,np.shape(pixel_pts)[0])] # reduce the density of the pixels
                    # pixel_pts = np.flip(pixel_pts, axis=1) # flip the order of the coordinates to x,y

                    # get 3d points from depth image
                    cloud_pts = self.get_cloud_pts(pixel_pts, data_pair['depth'])

                    # call tsl
                    self.new_states = False
                    self.pts_pub.publish(self.pts_to_pointcloud_msg(cloud_pts))
                    # wait for tsl publishes results
                    while not self.new_states and not rospy.is_shutdown():
                        rospy.sleep(0.1)
                    # states = rospy.wait_for_message(self.tsl_output_topic, PointCloud2)
                    # # plot results
                    # res = self.plot_results(cloud_pts, states)
                    # if res is not None:
                    #     cv2.imshow('plot', res)
                    #     res = cv2.cvtColor(res, cv2.COLOR_BGRA2BGR)
                    #     # self.video.write(cv2.resize(res, self.record_video_resolution))
                    #     key = cv2.waitKey(1)
                    #     if key == 27 or key == ord('q'):
                    #         print('terminating')
                    #         self.bag.close()
                    #         # self.video.release()
                    #         cv2.destroyAllWindows()
                    
                    # reset data_pair
                    data_pair['img'] = None
                    data_pair['depth'] = None
                    # cnt += 1
                    # if cnt == 1:
                    #     break
        except KeyboardInterrupt:
            print('terminated, saving the video file.')
            self.bag.close()
            # self.video.release()
            cv2.destroyAllWindows()

    def cb(self, msg):
        self.new_states = True

    def plt2np(self, fig):
        with io.BytesIO() as buff:
            fig.savefig(buff, format='raw')
            buff.seek(0)
            data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        return data.reshape((int(h), int(w), -1))

    def plot_results(self, cloud_pts, states):
        fig = plt.figure()
        plt.plot(cloud_pts[:,0], cloud_pts[:,1], 'g^', label="cloud")
        # plt.plot(self.ori_states[:,0], self.ori_states[:,1], 'bs', label="old states")
        # plt.plot(states[:,0], states[:,1], 'rs', label="new states")
        plt.plot(states[:,0], states[:,1], color = 'r', linewidth = '5', label="new states")
        # plt.plot(self.reference[:,1], self.reference[:,0], 'c*', label="target states")
        plt.xlabel('y')
        plt.ylabel('x')
        plt.legend(loc="upper left")
        # plt.axis([-0.15, 0.15, -0.15, 0.05])
        plt.gca().invert_yaxis()
        # plt.show()

        # plot = np.frombuffer(self.canvas.tostring_rgb(), dtype='uint8')
        plot = self.plt2np(fig)
        plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGRA)
        mask = cv2.cvtColor(resize(mask, height=np.shape(plot)[0]), cv2.COLOR_GRAY2BGRA)
        img = cv2.cvtColor(resize(img, height=np.shape(plot)[0]), cv2.COLOR_BGR2BGRA)
        res =  np.concatenate([plot, mask, img], axis=1)
        return res

    # def register(self, req):
    def register(self, img, mask, cloud_pts):
        # call registration service
        request = RegistrationServiceRequest()
        request.img = msgify(Image, img, 'bgr8')
        request.mask = msgify(Image, mask, 'mono8')
        request.cloud_points = Float64MultiArray()
        # cloud_pts[:,2]=1
        request.cloud_points.data = np.ravel(cloud_pts)
        response = self.registration_srv(request)
        states = np.reshape(response.states.data, (-1,3))
        print(states)

        if self.ori_states is not None:
            fig = plt.figure()
            plt.plot(cloud_pts[:,0], cloud_pts[:,1], 'g^', label="cloud")
            # plt.plot(self.ori_states[:,0], self.ori_states[:,1], 'bs', label="old states")
            # plt.plot(states[:,0], states[:,1], 'rs', label="new states")
            plt.plot(states[:,0], states[:,1], color = 'r', linewidth = '5', label="new states")
            # plt.plot(self.reference[:,1], self.reference[:,0], 'c*', label="target states")
            plt.xlabel('y')
            plt.ylabel('x')
            plt.legend(loc="upper left")
            plt.axis([-0.15, 0.15, -0.15, 0.05])
            plt.gca().invert_yaxis()
            # plt.show()

            # plot = np.frombuffer(self.canvas.tostring_rgb(), dtype='uint8')
            plot = self.plt2np(fig)
            plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGRA)
            mask = cv2.cvtColor(resize(mask, height=np.shape(plot)[0]), cv2.COLOR_GRAY2BGRA)
            img = cv2.cvtColor(resize(img, height=np.shape(plot)[0]), cv2.COLOR_BGR2BGRA)
            res =  np.concatenate([plot, mask, img], axis=1)
        else:
            res = None

        self.ori_states = states
        return res

    def get_cloud_pts(self, pixel_pts, depth):
        # pixel points to coordinates
        # camera_intrinsics = self.camera.read_camera_info()
        cloud_pts = read_points_from_region(pixel_pts, depth, region=3, camera_intrinsics=self.camera_intrinsics)
        # cloud_msg = rospy.wait_for_message(self.cloud_topic, PointCloud2)
        # cloud_pts = [self.get_xyz(pixel_pt, cloud_msg) for pixel_pt in pixel_pts] # sample_density*ndim
        # return np.array([cloud_pt[:2] for cloud_pt in cloud_pts if cloud_pt is not None]) # remove Nones
        return np.array([cloud_pt for cloud_pt in cloud_pts if not isnan(cloud_pt[0])]) # remove Nones

    # gete 3d point position from sensor_msgs point cloud
    def get_xyz(self, point_2d, cloud_msg):
        point_3d = read_point(point_2d, cloud_msg)
        return point_3d
        # return self.cam_to_rob(point_3d)
    
    def process_img(self, img):
        # img[img.shape[0]//2:,:,:] = 0 # delete the lower half
        # img[440:440+237, 557:557+151, :] = 0
        # img[270:270+154, 267:267+92, :] = 0
        return img

    def pts_to_pointcloud_msg(self, pts):
        # convert points to pointcloud message
        cloud_msg = PointCloud2()
        header = Header()
        header.frame_id = self.frame
        header.stamp = rospy.Time.now()
        cloud_msg.header = header
        # cloud_msg.header.stamp = rospy.Time.now()
        # cloud_msg.header.frame_id = self.frame
        cloud_msg.height = 1
        cloud_msg.width = len(pts)
        cloud_msg.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                            PointField('y', 4, PointField.FLOAT32, 1),
                            PointField('z', 8, PointField.FLOAT32, 1)]
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 12
        cloud_msg.row_step = 12*len(pts)
        cloud_msg.is_dense = False
        cloud_msg.data = np.array(pts).astype(np.float32).tostring()
        return cloud_msg

    # def cam_to_rob(self, point):
    #     return np.dot(self.cam_to_base, np.array([point[0], point[1], point[2], 1.0]))[:3]


if __name__ == '__main__' :
    rospy.init_node('bag_loader', anonymous=True)
    point_registration_node = PointRegistrationNode()
    # rospy.spin()