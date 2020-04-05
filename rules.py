from helpers import *

"""
Script that houses all the rules for - Squats, Pushups, Crunches

"""


class Squats():
        """
        Problems:
        - Squat depth
        - Tibia deviation
        - Straight back (Should not be bending forward)
        - Knees caving in (Ignore for POC since will need front side too, keep it simple and only use lateral side)

        SQUAT DEPTH:
        - Keypoints - L_hip, R_hip, L_knee, R_knee
        - Angles - Angle b/n hip-knee and hortizontal = angle_detected
        - Parameters
                - threshold = ? (15/20/30 degrees)
        - Equation -  if angle_detected > threshold:
                                                print('GO DEEPER') 

        """
        def __init__(self, pose, hip_angles, side='front'):
                self.kpt_names = {0: 'nose', 1: 'neck', 2: 'r_sho', 3: 'r_elb', 4: 'r_wri', 5: 'l_sho', 
                                        6: 'l_elb', 7: 'l_wri', 8: 'r_hip', 9: 'r_knee', 10: 'r_ank', 11: 'l_hip',
                                         12: 'l_knee', 13: 'l_ank', 14: 'r_eye', 15: 'l_eye', 16: 'r_ear', 17: 'l_ear'}

                self.pose = pose
                self.side = side
                self.corrections = {}

                self.hip_angles = hip_angles

                self.squat_depth_angle_threshold = 75
                self.tibia_deviation_angle_threshold = 70
                self.back_deviation_angle_threshold = 40

                self.rep_count = 0
                self.start = False
                self.down = True

                self.window = 40 # fix fps so window remains constant

        def check_if_going_down(self):

                # import pdb;pdb.set_trace()
                if self.hip_angles[-1] < self.hip_angles[-self.window]:
                        self.down = True
                else:
                        self.down = False
                return

        # Don't calculate for each frame - lowest angle should be < threshold 
        # Even standing position will be shown as bad form
        def squat_depth_angle(self):
                neck = get_coordinates(self.pose, 1)
                hip = average_coordinates(self.pose, 8, 11)
                knee = average_coordinates(self.pose, 9, 12)
                if neck and hip and knee:
                        hip_angle = calculate_angle(neck, hip, knee)
                else:
                        return -1  
                self.hip_angles.append(hip_angle)

                if hip_angle < 100:
                        self.start = True
                        print('STARRRT', len(self.hip_angles))
                else:
                        self.start = False


                if len(self.hip_angles) > self.window: 
                        self.check_if_going_down()

                if self.start == True and self.down == True:
                        if hip_angle < self.squat_depth_angle_threshold:  
                                return (hip_angle, 'good form', self.start, self.down)
                        else:
                                return (hip_angle, 'go deeper', self.start, self.down)
                else:
                        return (hip_angle, "Let's sqwat", self.start, self.down)


        def tibia_deviation(self):
                knee = average_coordinates(self.pose,9, 12)
                ankle = average_coordinates(self.pose, 10, 13)

                # Write conditions for when keypoints arent detected
                if knee and ankle:
                        tibia_angle = calculate_angle(knee, ankle)
                else:
                        return -1

                if self.start == True and self.down == True:
                        if tibia_angle < self.tibia_deviation_angle_threshold:
                                return (tibia_angle, 'knees before toes bro')
                        else:
                                return (tibia_angle, 'good form')
                else:
                        return (tibia_angle, "let's squat")


        def back_deviation(self):
                neck = get_coordinates(self.pose, 1)
                hip = average_coordinates(self.pose, 8, 11)

                if neck and hip:
                        back_angle = calculate_angle(neck, hip)
                else:
                        return -1

                if self.start == True and self.down == True:
                        if back_angle < self.back_deviation_angle_threshold:
                                return (back_angle, 'keep your back straighter')
                        else:
                                return (back_angle, 'good form')
                else:
                        return (back_angle, "let's squat")


        def all_corrections(self):
                self.corrections['squat_depth'] = self.squat_depth_angle()
                self.corrections['tibia_deviation'] = self.tibia_deviation()
                self.corrections['back_deviation'] = self.back_deviation()

                return self.corrections


        def rep_counter(self):
                """
                Store window between self.start = True and check if any angle 
                exists which is < hip_angle_threshold then it's a rep!
                """


class Pushups():
        """
        Problems:
        - Hip bending up
        - Hip bending down (Surya kriya style)
        - Lazy push up /Not going low enough
        - Hand should be away from head. 
    

        HIP BENDING:
        - Keypoints - Neck, Hip, knees 
        - Angles - hip angle
        - Parameters - 
        - Equation - 

        LAZY PUSH UP:
        - keypoints - Shoulder
        - Angles - 
        - Equations - 

        """
        def __init__(self, pose, elbow_angles, side='front'):
                self.kpt_names = {0: 'nose', 1: 'neck', 2: 'r_sho', 3: 'r_elb', 4: 'r_wri', 5: 'l_sho', 
                                        6: 'l_elb', 7: 'l_wri', 8: 'r_hip', 9: 'r_knee', 10: 'r_ank', 11: 'l_hip',
                                         12: 'l_knee', 13: 'l_ank', 14: 'r_eye', 15: 'l_eye', 16: 'r_ear', 17: 'l_ear'}

                self.pose = pose
                self.side = side
                self.corrections = {}

                self.elbow_angles = elbow_angles

                self.hip_bending_threshold = 160
                # self.surya_kriya_threshold = 160
                self.lazy_pushup_threshold = 90

                self.rep_count = 0
                self.start = False
                self.down = True

                self.window = 40 # fix fps so window remains constant


        # Merging surya kriya to this since they're essentially the same things
        def hip_bending(self):
                neck = get_coordinates(self.pose, 1)
                hip = average_coordinates(self.pose, 8, 11)
                knee = average_coordinates(self.pose, 9, 12)
                
                if neck and hip and knee:
                        hip_angle = calculate_angle(neck, hip, knee)  
                else:
                        return -1

                if hip_angle > self.hip_bending_threshold:
                        return (hip_angle, 'good form')
                else:
                        return (hip_angle, 'straighten your back')


        def check_if_going_down(self):
                if self.elbow_angles[-1] < self.elbow_angles[-self.window]:
                        self.down = True
                else:
                        self.down = False
                return


        def lazy_pushup(self):  
                shoulder = average_coordinates(self.pose, 2, 5)
                elbow = average_coordinates(self.pose, 3, 6)
                wrist = average_coordinates(self.pose, 4, 7)
                
                if shoulder and elbow and wrist:
                        elbow_angle = calculate_angle(shoulder, elbow, wrist)
                else:
                        return -1  
                self.elbow_angles.append(elbow_angle)

                if elbow_angle < 120:
                        self.start = True
                        # print('STARRRT', len(self.elbow_angles))

                else:
                        self.start = False

                if len(self.elbow_angles) > self.window: 
                        self.check_if_going_down()

                if self.start == True and self.down == True:
                        if elbow_angle < self.lazy_pushup_threshold:  
                                return (elbow_angle, 'Good form', self.start, self.down)
                        else:
                                return (elbow_angle, 'Go deeper', self.start, self.down)
                else:
                        return (elbow_angle, "Keep going!", self.start, self.down)


        def all_corrections(self):
                self.corrections['hip_bending'] = self.hip_bending()
                # self.corrections['surya_kriya'] = self.surya_kriya()
                self.corrections['lazy_pushup'] = self.lazy_pushup()

                return self.corrections



class Planks():
        """
        Problems:
        - Hip bending up
        - Hip bending down (Surya kriya style)

        HIP BENDING:
        - Keypoints - Neck, Hip, knees 
        - Angles - hip angle
        - Parameters - 
        - Equation - 

        """
        def __init__(self, pose, side='front'):
                self.kpt_names = {0: 'nose', 1: 'neck', 2: 'r_sho', 3: 'r_elb', 4: 'r_wri', 5: 'l_sho', 
                                        6: 'l_elb', 7: 'l_wri', 8: 'r_hip', 9: 'r_knee', 10: 'r_ank', 11: 'l_hip',
                                         12: 'l_knee', 13: 'l_ank', 14: 'r_eye', 15: 'l_eye', 16: 'r_ear', 17: 'l_ear'}

                self.pose = pose
                self.side = side
                self.corrections = {}

                self.hip_bending_threshold = 160


        def hip_bending(self):
                neck = get_coordinates(self.pose, 1)
                hip = average_coordinates(self.pose, 8, 11)
                knee = average_coordinates(self.pose, 9, 12)
                
                if neck and hip and knee:
                        hip_angle = calculate_angle(neck, hip, knee)  
                else:
                        return -1

                if hip_angle > self.hip_bending_threshold:
                        return (hip_angle, 'good form')
                else:
                        return (hip_angle, 'straighten your back')


        def all_corrections(self):
                self.corrections['hip_bending'] = self.hip_bending()

                return self.corrections
    
