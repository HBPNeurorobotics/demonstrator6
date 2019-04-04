# Imported Python Transfer Function
import numpy as np
@nrp.MapSpikeSink("motors", nrp.brain.motors, nrp.leaky_integrator_alpha)
@nrp.MapRobotPublisher('r_shoulder_yaw', Topic('/icub/r_shoulder_yaw/pos', std_msgs.msg.Float64))
@nrp.Neuron2Robot()
def swing(t, motors, r_shoulder_yaw):
    # clientLogger.info("Motor potential: {}".format(motors.voltage))
    r_shoulder_yaw.send_message(std_msgs.msg.Float64(30. * motors.voltage))
