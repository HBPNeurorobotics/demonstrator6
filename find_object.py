from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
@nrp.MapRobotSubscriber('camera',         Topic('/camera/image_raw',     Image))
@nrp.MapRobotPublisher( 'obj_pos_topic', Topic('/obj_pos', Point))
@nrp.Robot2Neuron()
def find_object(t, camera, obj_pos_topic):
    underSmpl = 5
    if int(t*50) % underSmpl == 0:
        import numpy as np
        from specs import localize_target
        from cv_bridge    import CvBridge
        import torch
        max_pix_value  = 1.0
        normalizer     = 255.0/max_pix_value
        cam_img = CvBridge().imgmsg_to_cv2(camera.value, 'rgb8')/normalizer
        cam_img = torch.tensor(cam_img).permute(2,1,0)

        targ = localize_target(cam_img)
        if targ[1] < 100: # otherwise it is detecting something else.
            targ = (-1, -1)

        msg = Point()
        msg.x = targ[0]
        msg.y = targ[1]
        obj_pos_topic.send_message(msg)