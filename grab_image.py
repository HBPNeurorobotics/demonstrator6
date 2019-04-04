# Imported Python Transfer Function
import numpy as np
import sensor_msgs.msg
from cv_bridge import CvBridge
@nrp.MapRobotSubscriber("camera", Topic("/icub_model/left_eye_camera/image_raw", sensor_msgs.msg.Image))
@nrp.MapSpikeSource("sensors", nrp.map_neurons(range(0, nrp.config.brain_root.n_sensors), lambda i: nrp.brain.sensors[i]), nrp.dc_source)
@nrp.MapVariable("last_mean_green", initial_value=None, scope=nrp.GLOBAL)
@nrp.Robot2Neuron()
def grab_image(t, camera, sensors, last_mean_green):
    # Take the image from the robot's left eye
    image_msg = camera.value
    if image_msg is not None:
        # Read the image into an array, mean over 3 colors, resize it for the network and flatten the result
        img = CvBridge().imgmsg_to_cv2(image_msg, "rgb8")
        green_channel = img[:,:,1]
        mean_green = np.mean(green_channel)
        if last_mean_green.value is None:
            last_mean_green.value = mean_green
        delta_mean_green = mean_green - last_mean_green.value
        # clientLogger.info("Delta mean green: {}".format(delta_mean_green))
        for neuron in sensors:
            neuron.amplitude = 3. * max(0., delta_mean_green) * np.random.rand()
        last_mean_green.value = mean_green
