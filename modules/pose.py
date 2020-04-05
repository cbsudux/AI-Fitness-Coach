import pprint

import cv2
import numpy as np

from modules.keypoints import BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS


class Pose(object):
    num_kpts = 18
    # kpt_names = ['nose', 'neck',
    #              'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
    #              'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
    #              'r_eye', 'l_eye',
    #              'r_ear', 'l_ear']
    kpt_names = {0: 'nose', 1: 'neck', 2: 'r_sho', 3: 'r_elb', 4: 'r_wri', 5: 'l_sho', 
                    6: 'l_elb', 7: 'l_wri', 8: 'r_hip', 9: 'r_knee', 10: 'r_ank', 11: 'l_hip',
                     12: 'l_knee', 13: 'l_ank', 14: 'r_eye', 15: 'l_eye', 16: 'r_ear', 17: 'l_ear'}

    sigmas = np.array([.26, .79, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35],
                      dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [0, 224, 255]

    def __init__(self, keypoints, confidence):
        super().__init__()
        self.kpt_names = Pose.kpt_names
        self.keypoints = keypoints
        self.confidence = confidence
        found_keypoints = np.zeros((np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32)
        found_kpt_id = 0
        for kpt_id in range(keypoints.shape[0]):
            if keypoints[kpt_id, 0] == -1:
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id]
            found_kpt_id += 1
        self.bbox = cv2.boundingRect(found_keypoints)
        self.id = None

    def update_id(self, id=None):
        self.id = id
        if self.id is None:
            self.id = Pose.last_id + 1
            Pose.last_id += 1

    def draw(self, img):
        assert self.keypoints.shape == (Pose.num_kpts, 2)

        pp = pprint.PrettyPrinter(indent=4)
        # import pdb; pdb.set_trace()

        pose_dict = {}
        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]
            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                cv2.circle(img, (int(x_a), int(y_a)), 3, (0,255,0), 3)
                # cv2.putText(img, self.kpt_names[kpt_a_id], (int(x_a), int(y_a)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                cv2.putText(img, str(kpt_a_id), (int(x_a), int(y_a)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                pose_dict[kpt_a_id] = [x_a,y_a]

            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]

            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                # print(kpt_b_id)
                cv2.circle(img, (int(x_b), int(y_b)), 3, (255,0,0), 3)
                # cv2.putText(img, self.kpt_names[kpt_b_id], (int(x_b), int(y_b)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                cv2.putText(img, str(kpt_b_id), (int(x_b), int(y_b)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                pose_dict[kpt_b_id] = [x_b,y_b]

            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), (0,0,0), 2)


    def get_pose(self):
        assert self.keypoints.shape == (Pose.num_kpts, 2)

        pose_dict = {}
        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]
            # print('kpt_a_id, global_kpt_a_id', kpt_a_id, global_kpt_a_id)
            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                pose_dict[kpt_a_id] = {'x' : x_a, 'y' : y_a}

            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]

            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                pose_dict[kpt_b_id] = {'x' : x_b, 'y' : y_b}

        return pose_dict


def get_similarity(a, b, threshold=0.5):
    num_similar_kpt = 0
    for kpt_id in range(Pose.num_kpts):
        if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1:
            distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)
            area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
            similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id]))
            if similarity > threshold:
                num_similar_kpt += 1
    return num_similar_kpt


def propagate_ids(previous_poses, current_poses, threshold=3):
    """Propagate poses ids from previous frame results. Id of pose is propagated,
    if there are at least `threshold` similar keypoints between pose from previous frame and current.

    :param previous_poses: poses from previous frame with ids
    :param current_poses: poses from current frame to assign ids
    :param threshold: minimal number of similar keypoints between poses
    :return: None
    """
    current_poses = sorted(current_poses, key=lambda pose: pose.confidence, reverse=True)  # match confident poses first
    mask = np.ones(len(previous_poses), dtype=np.int32)
    for current_pose_id in range(len(current_poses)):
        best_matched_id = None
        best_matched_pose_id = None
        best_matched_iou = 0
        for previous_pose_id in range(len(previous_poses)):
            if not mask[previous_pose_id]:
                continue
            iou = get_similarity(current_poses[current_pose_id], previous_poses[previous_pose_id])
            if iou > best_matched_iou:
                best_matched_iou = iou
                best_matched_pose_id = previous_poses[previous_pose_id].id
                best_matched_id = previous_pose_id
        if best_matched_iou >= threshold:
            mask[best_matched_id] = 0
        else:  # pose not similar to any previous
            best_matched_pose_id = None
        current_poses[current_pose_id].update_id(best_matched_pose_id)
