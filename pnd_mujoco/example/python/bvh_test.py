'''
Docstring for pnd_mujoco.example.python.bvh_test
'''
import time
import sys
import numpy as np

from bvhparser import * # pylint: disable=W0614,w0401

from pndbotics_sdk_py.core.channel import ChannelPublisher, ChannelFactoryInitialize # type: ignore # pylint: disable=E0401
from pndbotics_sdk_py.idl.default import adam_u_msg_dds__LowCmd_ # type: ignore  # pylint: disable=E0401
from pndbotics_sdk_py.idl.default import adam_u_msg_dds__LowState_ # type: ignore  # pylint: disable=E0401,w0611
from pndbotics_sdk_py.idl.adam_u.msg.dds_ import LowCmd_ # type: ignore  # pylint: disable=E0401
from pndbotics_sdk_py.idl.adam_u.msg.dds_ import LowState_ # type: ignore # pylint: disable=E0401,w0611
from pndbotics_sdk_py.idl.adam_u.msg.dds_ import HandCmd_ # type: ignore # pylint: disable=E0401
from pndbotics_sdk_py.idl.default import adam_u_msg_dds__HandCmd_ # type: ignore # pylint: disable=E0401


ADAM_U_NUM_MOTOR = 19
ADAM_MOTOR_DEF = [
    'waistRoll',
    'waistPitch',
    'waistYaw',
    'neckYaw',
    'neckPitch',
    'shoulderPitch_Left',
    'shoulderRoll_Left',
    'shoulderYaw_Left',
    'elbow_Left',
    'wristYaw_Left',
    'wristPitch_Left',
    'wristRoll_Left',
    'shoulderPitch_Right',
    'shoulderRoll_Right',
    'shoulderYaw_Right',
    'elbow_Right',
    'wristYaw_Right',
    'wristPitch_Right',
    'wristRoll_Right'
]
KP_CONFIG = [
    405.0,  # waistRoll (0)
    405.0,  # waistPitch (1)
    205.0,  # waistYaw (2)
    9.0,   # neckYaw (3)
    9.0,   # neckPitch (4)
    180.0,  # shoulderPitch_Left (5)
    180.0,   # shoulderRoll_Left (6)
    9.0,   # shoulderYaw_Left (7)
    9.0,   # elbow_Left (8)
    9.0,   # wristYaw_Left (9)
    9.0,   # wristPitch_Left (10)
    9.0,   # wristRoll_Left (11)
    180.0,  # shoulderPitch_Right (12)
    180.0,   # shoulderRoll_Right (13)
    9.0,   # shoulderYaw_Right (14)
    9.0,   # elbow_Right (15)
    9.0,   # wristYaw_Right (16)
    9.0,   # wristPitch_Right (17)
    9.0    # wristRoll_Right (18)
]

# Kd 配置数组（对应19个关节）
KD_CONFIG = [
    6.75,   # waistRoll (0)
    6.75,   # waistPitch (1)
    3.42,   # waistYaw (2)
    0.9,   # neckYaw (3)
    0.9,   # neckPitch (4)
    1.8,   # shoulderPitch_Left (5)
    1.8,   # shoulderRoll_Left (6)
    0.9,   # shoulderYaw_Left (7)
    0.9,   # elbow_Left (8)
    0.9,   # wristYaw_Left (9)
    0.9,   # wristPitch_Left (10)
    0.9,   # wristRoll_Left (11)
    1.8,   # shoulderPitch_Right (12)
    1.8,   # shoulderRoll_Right (13)
    0.9,   # shoulderYaw_Right (14)
    0.9,   # elbow_Right (15)
    0.9,   # wristYaw_Right (16)
    0.9,   # wristPitch_Right (17)
    0.9    # wristRoll_Right (18)
]


open_arm_pos = np.array([0, 0, 0,
                       0.7, -0.5,
        -1.6, 2.06, -1.65, -1.77,
                      0.32, 0, 0,
       - 1.6, -2.06, 1.65, -1.77,
                      0.32, 0, 0
],
                              dtype=float)

close_arm_pos = np.array([0, 0, 0,
                             0, 0,
                       0, 0, 0, 0,
                          0, 0, 0,
                       0, 0, 0, 0,
                          0, 0, 0
],
                                dtype=float)

open_hand = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000], dtype=int)
close_hand = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int)

dt = 0.010416


# input("Press enter to start")

urdf_to_bvh = {
    'waistRoll' : ('Spine1', 2),
    'waistPitch': ('Spine1', 1),
    'waistYaw': ('Spine1', 0),

    'neckYaw': ('Neck1', 0),
    'neckPitch': ('Neck1', 1),

    'shoulderPitch_Left': ('LeftArm', 1),
    'shoulderRoll_Left': ('LeftArm', 2),
    'shoulderYaw_Left': ('LeftArm', 0),
    'elbow_Left': ('LeftForeArm', 1), # elbow is pitch

    'wristYaw_Left': ('LeftHand', 0),
    'wristPitch_Left': ('LeftHand', 1),
    'wristRoll_Left': ('LeftHand', 2),
    
    'shoulderPitch_Right': ('RightArm', 1),
    'shoulderRoll_Right': ('Spine1', 2),
    'shoulderYaw_Right': ('RightArm', 0),
    'elbow_Right': ('RightForeArm', 1),

    'wristYaw_Right': ('RightHand', 0),
    'wristPitch_Right': ('RightHand', 1),
    'wristRoll_Right': ('RightHand', 2)
}

def importbvh():
    '''
    Docstring for importbvh
    '''
    # --- BVH → URDF mapping ---
    # xyz is yxz in bhv
    bvh_to_urdf = {
        'Spine':   ('waistYaw',   'Z'),
        'Spine1':  ('waistPitch', 'Y'),
        'Spine2':  ('waistRoll',  'X'),

        'Neck':   ('neckYaw',   'Z'),
        'Neck1':  ('neckPitch', 'Y'),

        'LeftShoulder': ('shoulderYaw_Left',   'Z'),
        'LeftArm':      ('shoulderPitch_Left', 'Y'),
        'LeftForeArm':  ('elbow_Left',          'Y'),
        'LeftHand':     ('wristPitch_Left',     'Y'),

        'RightShoulder': ('shoulderYaw_Right',   'Z'),
        'RightArm':      ('shoulderPitch_Right', 'Y'),
        'RightForeArm':  ('elbow_Right',          'Y'),
        'RightHand':     ('wristPitch_Right',     'Y'),
    }


    filename = 'example/python/take002_chr01.bvh'
    skeleton_data = ProcessBVH(filename)
    joints = skeleton_data[0]
    joints_offsets = skeleton_data[1]
    joints_hierarchy = skeleton_data[2]
    root_positions = skeleton_data[3]
    joints_rotations = skeleton_data[4] #this contains the angles in degrees, size = num frames
    joints_saved_angles = skeleton_data[5] #this contains channel information. E.g ['Xrotation', 'Yrotation', 'Zrotation']
    joints_positions = skeleton_data[6]
    joints_saved_positions = skeleton_data[7]


    """
    Number of frames skipped is controlled with this variable below. If you want all frames, set to 1.
    """
    frame_skips = 1

    frames = []

    for i in range(0,len(joints_rotations), frame_skips):

        frame_data = joints_rotations[i]

        frame_joints_rotations = {}

        # for joint in joints:
        #     if joint in bvh_to_urdf:
        #         frame_joints_rotations[joint] = []

        #fill in the rotations dict
        joint_index = 0
        for joint in joints:
            if joint in bvh_to_urdf:
                frame_joints_rotations[joint] = frame_data[joint_index:joint_index+3]

            joint_index += 3
        frames.append(frame_joints_rotations)
            
    print('frame: ', frames[1])
    return frames

def test_main():
    frames = importbvh()

    runing_time = 0.0
    # Create a publisher to publish the data defined in UserData class
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    cmd = adam_u_msg_dds__LowCmd_()

    hand_pub = ChannelPublisher("rt/handcmd", HandCmd_)
    hand_pub.Init()
    hand_cmd = adam_u_msg_dds__HandCmd_()


# 96 fps
    frame_duration = 0.01041667 # a frame should take this much time
    for frame in frames:
        frame_start = time.perf_counter() # float value of time in seconds

        for i in range(ADAM_U_NUM_MOTOR):
            if ADAM_MOTOR_DEF[i] in urdf_to_bvh:
                angle_deg = (frame[urdf_to_bvh[ADAM_MOTOR_DEF[i]][0]][urdf_to_bvh[ADAM_MOTOR_DEF[i]][1]])
                print(ADAM_MOTOR_DEF[i], angle_deg)
                cmd.motor_cmd[i].q = math.radians(angle_deg)
            else:
                cmd.motor_cmd[i].q = 0

            cmd.motor_cmd[i].kp = KP_CONFIG[i]
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = KD_CONFIG[i]
            cmd.motor_cmd[i].tau = 0.0

        for i in range(12):
            hand_cmd.position[i] = open_hand[i]

        pub.Write(cmd)
        hand_pub.Write(hand_cmd)

        time_until_next_frame = frame_duration - (time.perf_counter() - frame_start)
        if time_until_next_frame > 0:
            time.sleep(time_until_next_frame)


def pnd_main():
    runing_time = 0.0
    # Create a publisher to publish the data defined in UserData class
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    cmd = adam_u_msg_dds__LowCmd_()

    hand_pub = ChannelPublisher("rt/handcmd", HandCmd_)
    hand_pub.Init()
    hand_cmd = adam_u_msg_dds__HandCmd_()
    #for i in range(19):
     #   cmd.motor_cmd[i].q = 0.0
      #  cmd.motor_cmd[i].kp = 0.0
       # cmd.motor_cmd[i].dq = 0.0
        #cmd.motor_cmd[i].kd = 0.0
        #cmd.motor_cmd[i].tau = 0.0

    while True:
        step_start = time.perf_counter()

        runing_time = runing_time + dt

        if (runing_time > 10.0):
            return

        if (runing_time < 3.0):
            # Stand up in first 3 second
            
            # Total time for standing up or standing down is about 1.2s
            phase = np.tanh(runing_time / 1.2)
            for i in range(ADAM_U_NUM_MOTOR):
                cmd.motor_cmd[i].q = phase * open_arm_pos[i] + (
                    1 - phase) * close_arm_pos[i]
                # 使用配置的 Kp 和 Kd 值
                cmd.motor_cmd[i].kp = KP_CONFIG[i]
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].kd = KD_CONFIG[i]
                cmd.motor_cmd[i].tau = 0.0

            for i in range(12):
                hand_cmd.position[i] = close_hand[i]
        else:
            phase = np.tanh((runing_time - 3.0) / 1.2)
            for i in range(ADAM_U_NUM_MOTOR):
                cmd.motor_cmd[i].q = phase * close_arm_pos[i] + (
                    1 - phase) * open_arm_pos[i]
                # 使用配置的 Kp 和 Kd 值
                cmd.motor_cmd[i].kp = KP_CONFIG[i]
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].kd = KD_CONFIG[i]
                cmd.motor_cmd[i].tau = 0.0

            for i in range(12):
                hand_cmd.position[i] = open_hand[i]

        #print(cmd.motor_cmd[6].q)
        pub.Write(cmd)
        hand_pub.Write(hand_cmd)

        time_until_next_step = dt - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

if __name__ == '__main__':

    if len(sys.argv) <2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(1, sys.argv[1])

    # pnd_main()

    test_main()
