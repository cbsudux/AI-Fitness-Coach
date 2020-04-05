from helpers import *

"""
Get's pose and figures out the angles that have the biggest derivatives

"""

"""
Finding activity:
- Find the angle that changes the most (Pushup --> elbows and shoulder angle | Squats --> Hip and knee angle)
- 4 angles : Shoulder | Elbows | Hip | Knees
- Find the derivative of the angle that changes the most and figure out activity based on that
- Keep getting pose and if activity  = False: --> continue


"""
            
# list_of_elbow_angles = []
list_of_shoulder_angles = []
list_of_hip_angles = []
list_of_knee_angles = []

def get_activity(pose, list_of_elbow_angles):
  
    ############# These should be initialized in demo.py ########################
    
    # Find a window size that matches squats and pushups (16 is too small for squats and > 30 is too big for pushups)
    # You can go over self.start, but not over down = True. (else rep will  not be counted)
    # So max_limit for window is till down ends

    window_size = 16

    # Get the coordinates
    shoulder = average_coordinates(pose, 2, 5)
    elbow = average_coordinates(pose, 3, 6)
    wrist = average_coordinates(pose, 4, 7)
    hip = average_coordinates(pose, 8, 11)
    knee = average_coordinates(pose, 9, 12)
    ankle = average_coordinates(pose, 10, 13)
    neck = get_coordinates(pose, 1)

    if elbow and wrist and neck and knee and ankle and hip:
        # Get the angles
        elbow_angle = calculate_angle(wrist, elbow, shoulder)
        shoulder_angle = calculate_angle(elbow, shoulder, hip)
        hip_angle = calculate_angle(neck, hip, knee)
        knee_angle = calculate_angle(hip, knee, ankle)
    else:
        print("Skipping Frame")
        return None, list_of_elbow_angles

    #print("Angle Computation Successful")
    # Keep track of the major angles
    list_of_elbow_angles.append(elbow_angle)
    list_of_shoulder_angles.append(shoulder_angle)
    list_of_hip_angles.append(hip_angle)
    list_of_knee_angles.append(knee_angle)
    # print('list_of_elbow_angles',list_of_elbow_angles)

    # Get the length of the tracked frames
    # Any of the angle lists could be used as all would have the same size
    track_size = len(list_of_hip_angles)         
    print("Track size is:", track_size)
    # print (list_of_hip_angles[-1])

    # Check for enough values to compute the derivative
    if track_size == window_size:
        # Compute the derivatives
        d_elbow = (list_of_elbow_angles[-1] - list_of_elbow_angles[-window_size])/track_size
        d_shoulder = (list_of_shoulder_angles[-1] - list_of_shoulder_angles[-window_size])/track_size
        d_hip = (list_of_hip_angles[-1] - list_of_hip_angles[-window_size])/track_size
        d_knee = (list_of_knee_angles[-1] - list_of_knee_angles[-window_size])/track_size
        # Rules for figuring out the activity ( Squats | Pushup | Plank )
        print("d_hip", d_hip)
        print("d_elbow", d_elbow) 
        print("d_knee", d_knee)
        print("d_shoulder", d_shoulder)
        # Squats ( hip_angle )
        #if abs(d_hip) > abs(d_elbow) or abs(d_hip) > abs(d_shoulder) or abs(d_knee) > abs(d_elbow):
        if abs(d_hip) > abs(d_shoulder) or abs(d_knee) > abs(d_elbow):
            return "Squats"
        
        # Pushup ( elbow angle and shoulder angle)
        elif abs(d_elbow) > abs(d_hip) and abs(d_shoulder) > abs(d_knee):
            return "Pushups", list_of_elbow_angles

        # Plank
        else:
            return "Planks"


    else:
        return None, list_of_elbow_angles


