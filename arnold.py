from rules import Squats


def arnold(pose, activity):
"""
Function that takes the pose and activity and returns if correct form. 
"""

# Insert trainer code here

    if activity == 'Squats':
        squats = Squats(final_pose, list_of_hip_angles)
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