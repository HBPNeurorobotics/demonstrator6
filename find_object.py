from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
@nrp.MapRobotSubscriber('camera',         Topic('/camera/image_raw',     Image))
@nrp.MapRobotPublisher( 'obj_pos_topic', Topic('/obj_pos', Point))
@nrp.MapRobotPublisher( 'plot_topic',     Topic('/obj_pos_img',            Image))
@nrp.Robot2Neuron()
def find_object(t, camera, obj_pos_topic, plot_topic):
    underSmpl = 5
    if int(t*50) % underSmpl == 0:
        import numpy as np
        from specs import localize_target, mark_target
        from cv_bridge    import CvBridge
        import torch
        max_pix_value  = 1.0
        normalizer     = 255.0/max_pix_value
        cam_img = CvBridge().imgmsg_to_cv2(camera.value, 'rgb8')/normalizer
        cam_img = torch.tensor(cam_img).permute(2,1,0)

        targ = localize_target(cam_img)
        if targ[1] < 150 or targ[1] > 200: # otherwise it is detecting something not on the belt.
            targ = (-1, -1)
        #clientLogger.info(targ)
        msg = Point()
        msg.x = targ[0]
        msg.y = targ[1]
        obj_pos_topic.send_message(msg)
        
        # Draw box and publish that
        if targ[0] > 95 and targ[0] < 105:
            cam_img = mark_target(cam_img, targ)
        img_msg = cam_img.numpy().transpose(2,1,0)*255.0
        plot_topic .send_message(CvBridge().cv2_to_imgmsg(img_msg.astype(np.uint8),'rgb8'))