import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from bvh import Bvh
import numpy as np

# --- Load BVH ---
with open("take001_chr00.bvh") as f:
    mocap = Bvh(f.read())

n_frames = mocap.nframes
frame_time = float(mocap.frame_time)

# --- URDF limits (upper body only) ---
urdf_limits = {
    'waistRoll': (-0.279, 0.279),
    'waistPitch': (-0.663, 1.361),
    'waistYaw': (-0.829, 0.829),

    'shoulderPitch_Left': (-3.613, 2.042),
    'shoulderRoll_Left': (-0.628, 2.793),
    'shoulderYaw_Left': (-2.583, 2.583),
    'elbow_Left': (-2.496, 0.209),
    'wristPitch_Left': (-0.96, 0.96),

    'shoulderPitch_Right': (-3.613, 2.042),
    'shoulderRoll_Right': (-2.793, 0.628),
    'shoulderYaw_Right': (-2.583, 2.583),
    'elbow_Right': (-2.496, 0.209),
    'wristPitch_Right': (-0.96, 0.96),

    'neckYaw': (-1.571, 1.571),
    'neckPitch': (-0.873, 0.873),
}

# --- BVH → URDF mapping ---
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

AXIS_INDEX = {'Y': 0, 'X': 1, 'Z': 2}

def get_rot(frame, joint, axis_letter):
    channels = mocap.joint_channels(joint)

    # Find index of requested rotation axis
    for i, ch in enumerate(channels):
        if axis_letter.lower() in ch.lower():
            values = mocap.frame_joint_channels(frame, joint, channels)
            return np.deg2rad(float(values[i]))

    return 0.0

class BVHPublisher(Node):
    def __init__(self):
        super().__init__('bvh_to_jointstate')
        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(frame_time, self.tick)
        self.frame = 0

    def tick(self):
        if self.frame >= n_frames:
            rclpy.shutdown()
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        for bvh_joint, (urdf_joint, axis) in bvh_to_urdf.items():
            val = get_rot(self.frame, bvh_joint, axis)
            if urdf_joint in urdf_limits:
                lo, hi = urdf_limits[urdf_joint]
                val = np.clip(val, lo, hi)
            msg.name.append(urdf_joint)
            msg.position.append(val)

        self.pub.publish(msg)
        self.frame += 1

def main():
    rclpy.init()
    node = BVHPublisher()
    rclpy.spin(node)

if __name__ == "__main__":
    main()
