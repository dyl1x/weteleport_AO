import time
from pndbotics_sdk_py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from pndbotics_sdk_py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from pndbotics_sdk_py.idl.default import adam_u_msg_dds__LowCmd_
from pndbotics_sdk_py.idl.default import adam_u_msg_dds__LowState_
from pndbotics_sdk_py.idl.adam_u.msg.dds_ import LowCmd_
from pndbotics_sdk_py.idl.adam_u.msg.dds_ import LowState_


def LowStateHandler(msg: LowState_):
    print("IMU state: ", msg.imu_state)
    # print("motor[0] state: ", msg.motor_state[0])


if __name__ == "__main__":
    ChannelFactoryInitialize(1, "lo")
    low_state_suber = ChannelSubscriber("rt/lowstate", LowState_)

    low_cmd_puber = ChannelPublisher("rt/lowcmd", LowCmd_)
    low_cmd_puber.Init()

    cmd = adam_u_msg_dds__LowCmd_()
    for i in range(19):
        cmd.motor_cmd[i].q= 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0
        
    while True:
        for i in range(19):
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].dq = 0.0 
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].tau = 1.0 
        
        #Publish message
        low_cmd_puber.Write(cmd)
        time.sleep(0.002)
