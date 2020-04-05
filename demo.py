import argparse
import imutils

import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, propagate_ids
from val import normalize, pad_width

from rules import Squats, Pushups, Planks
from helpers import *
from activity_recognizer import get_activity


class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def run_demo(net, image_provider, height_size, cpu, track_ids):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    
    list_of_elbow_angles = []
    list_of_shoulder_angles = []
    list_of_hip_angles = []
    list_of_knee_angles = []
    
    rep_count = 0
    rep_not_counted = True
    rep_start = False
    rep_end = False
    
    activity = None

    rotate = True
    rotate_90 = True
    rotate_270 = False

    for img in image_provider:
        if rotate:
            if rotate_90:
                img = imutils.rotate_bound(img, 90)
            if rotate_270:
                img = imutils.rotate_bound(img, 270)

        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True) # pose entries -> Array showing number of poses, and which keypoint in each pose
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []

        # do not iterate over all poses, instead only choose the one with largest bounding box (pose.bbox)
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            # import pdb;pdb.set_trace()
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

            final_pose = pose.get_pose()
            draw(final_pose, img, names = False, lines = True, angles = [5, 6, 7])

            # """
            # - Find out sleeping position (or not standing? Easier to write rules for standing)
            # - 
            # """

            # Figure out if standing or sleeping
            neck = get_coordinates(final_pose, 1)
            hip = average_coordinates(final_pose, 8, 11)
            knee = average_coordinates(final_pose, 9, 12)

            # checking if standing by comparing 'y' coordinates
            # neck is not good. Choose something which is below hips in sleeping but not in standing
            if neck[1] > hip[1] and hip[1] > knee[1]:
                rotate = False
            else: 
                rotate = True
                # checking which side to rotate by comparing x coordinates
                print('neck[0], hip[0], neck[1], hip[1]',neck[0], hip[0], neck[1], hip[1])
                if neck[0] < hip[0]:
                    rotate_90 = True
                    rotate_270 = False
                else:
                    rotate_90 = False
                    rotate_270 = True
                # print('rotate, rotate_90, rotate_270',rotate, rotate_90, rotate_270)
                continue    

            if activity == None:
                activity, list_of_elbow_angles = get_activity(final_pose, list_of_elbow_angles)
                continue
            else:
                pass

            pushups = Pushups(final_pose, list_of_elbow_angles)

            try:
                corrections = pushups.all_corrections()
                for problem, correction in corrections.items():
                    if problem == 'lazy_pushup':
                        list_of_elbow_angles.append(correction[0])
                        
                        # if rep_not_counted:
                        start = correction[2]
                        down = correction[3]

                        # print('1] start, down, rep_start, rep_end, rep_count, elbow angle', \
                        #     start, down, rep_start, rep_end, rep_count, correction[0])

                        print('Activity : {} ||| Correction : {} ||| Rep count : {}'.format(activity, correction[1], rep_count), end = '\r')
                        # print('Correction : {} ||| Rep count : {}'.format(correction[1], rep_count), end = '\r')

                        if start == True and down == True and rep_start == False:
                            start_index = len(list_of_elbow_angles) - 1
                            rep_start = True
                            rep_end = False
                            rep_counted = False 

                        if start == False and down == False and rep_end == False:
                            stop_index = len(list_of_elbow_angles) - 1
                            interval = []

                            # import pdb; pdb.set_trace()
                            for angle in list_of_elbow_angles[start_index: stop_index]:
                                if angle < pushups.lazy_pushup_threshold and not rep_counted:
                                    rep_count = rep_count + 1
                                    rep_counted = True
                            rep_end = True
                            rep_start = False

            squats = Squats(final_pose, list_of_hip_angles)
            print("Ran Squats")
            
            try:
                corrections = squats.all_corrections()
                for problem, correction in corrections.items():
                    if problem == 'squat_depth':
                        list_of_hip_angles.append(correction[0])
                       
                        # if rep_not_counted:
                        start = correction[2]
                        down = correction[3]

                        print('1] start, down, rep_start, rep_end, rep_count, squat angle', \
                            start, down, rep_start, rep_end, rep_count, correction[0])

                        if start == True and down == True and rep_start == False:
                            start_index = len(list_of_hip_angles) - 1
                            rep_start = True
                            rep_end = False
                            rep_counted = False 

                        if start == False and down == False and rep_end == False:
                            stop_index = len(list_of_hip_angles) - 1
                            interval = []

                            # import pdb; pdb.set_trace()
                            for angle in list_of_hip_angles[start_index: stop_index]:
                                if angle < squats.squat_depth_angle_threshold and not rep_counted:
                                    rep_count = rep_count + 1
                                    rep_counted = True
                            rep_end = True
                            rep_start = False

                        print('2] start, down, rep_start, rep_end, rep_count, squat angle', \
                            start, down, rep_start, rep_end, rep_count, correction[0])
                               
            except:
                pass


        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        if track_ids == True:
            propagate_ids(previous_poses, current_poses)
            previous_poses = current_poses
            for pose in current_poses:
                cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                              (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        # cv2.imwrite('output.jpg', img)
        key = cv2.waitKey(1)
        if key == 27:  # esc
           return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track-ids', default=False, help='track poses ids')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    frame_provider = ImageReader(args.images)
    if args.video != '':
        frame_provider = VideoReader(args.video)

    # Calling a function once and running a loop inside it is good practice. Do not put the function call
    # inside a loop and call the function over and over
    run_demo(net, frame_provider, args.height_size, args.cpu, args.track_ids)
