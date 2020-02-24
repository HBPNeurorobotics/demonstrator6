#!/usr/bin/env python
""" """

__author__ = 'Alban Bornet'

import rospy
import os
import torch
import torch.nn.functional as F
import numpy as np
from external_module_interface.external_module import ExternalModule
from sensor_msgs.msg import Image
from std_msgs.msg    import Float32MultiArray
from prednet         import PredNet
from cv_bridge       import CvBridge
from std_msgs.msg    import Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from specs           import localize_target, complete_target_positions, mark_target, exp_dir


class PredictiveCoding(ExternalModule):

    def __init__(self, module_name=None, steps=1, underSmpl=5, nt=15, t_extrap=5, n_feat=1, max_pix_value=1.0,
        C_channels=3, scale=4, use_new_w=False, use_trained_w=True, do_train=False, lr=1e-4, epoch_loop=100):
        super(PredictiveCoding, self).__init__(module_name, steps)

        # Subscribers
        self.camera     = None
        self.camera_sub = rospy.Subscriber("chatter", Image, self.camera_sub_callback)

        # Publishers
        self.plot_pub     = rospy.Publisher('plot_topic',     Image,             queue_size=1)
        self.latent_pub   = rospy.Publisher('latent_topic',   Float32MultiArray, queue_size=1)
        self.pred_pos_pub = rospy.Publisher('pred_pos_topic', Float32MultiArray, queue_size=1)

        # Image and model parameters
        self.underSmpl      = underSmpl      # Avoiding too sharp time resolution (no change between frames)
        self.nt             = nt             # Number of "past" frames given to the network
        self.t_extrap       = t_extrap       # After this frame, input is not used for future predictions
        self.n_feat         = n_feat         # Factor for number of features used in the network
        self.max_pix_value  = max_pix_value  # Depends on what's inside the PredNet code
        self.normalizer     = 255.0/self.max_pix_value
        self.C_channels     = C_channels     # 1 or 3 (number of color channels)
        self.A_channels     = (self.C_channels, self.n_feat*4, self.n_feat*8, self.n_feat*16)
        self.R_channels     = (self.C_channels, self.n_feat*4, self.n_feat*8, self.n_feat*16)
        self.scale          = scale          # 2 or 4 (how much layers down/upsample images)
        self.pad            = 8 if self.scale == 4 else 0  # For up/downsampling to work
        self.model_name     = 'model' + str(self.n_feat)+'.pt'
        self.new_model_path = os.getcwd() + '/resources/' + self.model_name
        self.trained_w_path = exp_dir + self.model_name    # exp_dir computed in specs.py 
        self.device         = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        # Training parameters
        self.use_new_w     = use_new_w      # If True, do not use weights that are saved in new_model_path
        self.use_trained_w = use_trained_w  # If above is False, use trained_w_path as model weights
        self.do_train      = do_train       # Train with present frames if True, predicts future if False
        self.initial_lr    = lr             # Then, the learning rate is scheduled with cosine annealing
        self.epoch_loop    = epoch_loop     # Every epoch_loop, a prediction is made, to monitor progress

        # Variables that can change over time
        self.pred_msg     = None
        self.model        = None
        self.model_path   = None
        self.model_inputs = None
        self.optimizer    = None
        self.scheduler    = None
        self.running_step = 0

        self.last_cam_time = rospy.Time.now().to_sec()*1000


    def camera_sub_callback(self, data):
        self.camera = data

    
    def run_step(self):

        # Check that the camera device is on and that it is the right time-step
        if self.camera is not None:
            t = self.camera.header.stamp.to_secs()*1000.0   # in milliseconds
            if t > self.last_cam_time + 20*self.underSmpl:           # one ros time-step is 20 ms
                self.last_cam_time = t

                # Collect input image and initialize the network input
                cam_img = CvBridge().imgmsg_to_cv2(self.camera, 'rgb8')/self.normalizer
                if self.C_channels == 3:  # Below I messed up, it should be (2,0,1) but the model is already trained.
                    cam_img = torch.tensor(cam_img, device=self.device).permute(2,1,0)  # --> channels last
                if self.C_channels == 1:
                    cam_img = cam_img[:,:,1]  # .mean(axis=2)
                    cam_img = torch.tensor(cam_img, device=self.device).unsqueeze(dim=2).permute(2,1,0)
                img_shp = cam_img.shape
                cam_inp = F.pad(cam_img, (self.pad, self.pad), 'constant', 0.0)  # width may need to be 256
                if self.model_inputs is None:
                    self.model_inputs = torch.zeros((1,self.nt)+cam_inp.shape, device=self.device)

                # Update the model or the mode, if needed
                self.running_step = self.running_step + 1
                if self.new_model_path != self.model_path:

                    # Update the model path if new or changed and reset prediction plot
                    self.model_path = self.new_model_path
                    self.pred_msg   = torch.ones(img_shp[0], img_shp[1]*(self.nt-self.t_extrap+1), img_shp[2]+10)*64.0

                    # Load or reload the model
                    self.model = PredNet(self.R_channels, self.A_channels, device=self.device, t_extrap=self.t_extrap, scale=self.scale)
                    if self.device == 'cuda': self.model = self.model.to('cuda')
                    if self.running_step == 1:
                        try:
                            if self.use_new_w:
                                a = 1./0.
                            if self.use_trained_w:
                                self.model.load_state_dict(torch.load(self.trained_w_path))
                                rospy.loginfo('Model initialized with pre-trained weights.')
                            else:
                                self.model.load_state_dict(torch.load(self.model_path))
                                rospy.loginfo('Learning weights loaded in the model.')
                        except:
                            rospy.loginfo('No existing weight file found. Model initialized randomly.')
                    
                # Initialize some variables needed for training
                time_loss_w = [1.0/(self.nt-1) if s > 0 else 0.0 for s in range(self.nt)]
                if self.t_extrap < self.nt:
                    time_loss_w = [w if n < self.t_extrap else 2.0*w for n, w in enumerate(time_loss_w)]

                # Initialize the optimizer and the scheduler if needed
                if None in [self.optimizer, self.scheduler]:
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initial_lr)
                    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=50)

                # Save the model at each epoch
                if self.running_step % self.epoch_loop == 1:
                    torch.save(self.model.state_dict(), self.model_path)

                # Check that the model exists and initialize plot message
                if self.model is not None:

                    # Feed network and train it or compute prediction
                    self.model_inputs = self.model_inputs.roll(-1, dims=1)
                    self.model_inputs[0,-1,:,:,:] = cam_inp
                    if self.running_step > self.nt:

                        # Compute prediction along present frames and updates weights
                        if self.do_train:

                            # Compute prediction loss for every frame
                            pred, latent = self.model(self.model_inputs, self.nt)
                            loss         = torch.tensor([0.0], device=self.device)
                            for s in range(self.nt):
                                error = (pred[s][0] - self.model_inputs[0][s])**2
                                loss += torch.sum(error)*time_loss_w[s]

                            # Backward pass and weight updates
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                            self.scheduler.step()

                        # Predicts future frames without weight updates
                        else:
                            with torch.no_grad():
                                pred, latent = self.model(self.model_inputs[:,-self.t_extrap:,:,:,:], self.nt)

                        # Collect prediction frames
                        displays = [cam_img]                   # First frame to be displayed is the present frame
                        targ_pos = [localize_target(cam_img)]  # Localize the target on the present camera frame
                        t_stamps = [t]                         # Time of the present frame is the camera rostime
                        for s in range(self.nt-self.t_extrap):
                            disp = torch.detach(pred[self.t_extrap+s].clamp(0.0, 1.0)[0,:,:,self.pad:-self.pad]).cpu()
                            targ_pos.append(localize_target(disp))
                            displays.append(disp)
                            t_stamps.append(t + (s+1)*0.02*self.underSmpl)  # Not sure about this

                        # Complete for missing target positions, highlight target and set the display message
                        if 0 < np.sum([any([np.isnan(p) for p in pos]) for pos in targ_pos]) < len(targ_pos)-2:
                            targ_pos = complete_target_positions(targ_pos)
                        for s, (disp, pos) in enumerate(zip(displays, targ_pos)):
                            self.pred_msg[:,s*img_shp[1]:(s+1)*img_shp[1],:img_shp[2]] = mark_target(disp, pos)

                        # Print loss or prediction messages
                        if self.do_train:
                            rospy.loginfo('Epoch: %2i - step: %2i - error: %5.4f - lr: %5.4f' % (int(self.running_step/self.epoch_loop), self.running_step % self.epoch_loop, loss.item(), self.scheduler.get_lr()[0]))
                        else:
                            rospy.loginfo('Prediction for future target locations: ' + str(targ_pos))

                        # Send latent state message (latent[0] to remove batch dimension)
                        latent_msg = list(latent[0].cpu().numpy().flatten())
                        layout_msg = MultiArrayLayout(dim=[MultiArrayDimension(size=d) for d in latent[0].shape])
                        self.latent_pub.publish(Float32MultiArray(layout=layout_msg, data=latent_msg))

                        # Send predicted position according to the index of the frame that has to be reported
                        pos_3d_msg = [[1.562-p[0]/156.274, -0.14-p[1]/152.691, 0.964+p[0]-p[0]] for p in targ_pos]
                        pos_4d_msg = [[p[0], p[1], p[2], s] for (s, p) in zip(t_stamps, pos_3d_msg)]  # Add time stamps
                        pos_4d_msg = [p for pos in pos_3d_msg for p in pos]                           # Flatten the list
                        layout_msg = MultiArrayLayout(dim=[MultiArrayDimension(size=d) for d in [len(targ_pos), 4]])
                        self.pred_pos_pub.publish(Float32MultiArray(layout=layout_msg, data=pos_3d_msg))

                    # Collect input frames
                    inpt_msg = torch.zeros(img_shp[0], img_shp[1]*(self.nt-self.t_extrap+1), img_shp[2])
                    for s in range(self.nt-self.t_extrap):
                        inpt_msg[:,(s+1)*img_shp[1]:(s+2)*img_shp[1],:] = self.model_inputs[0,self.t_extrap+s,:,:,self.pad:-self.pad]

                    # Build and send the display message
                    plot_msg = torch.cat((self.pred_msg, inpt_msg), 2).numpy().transpose(2,1,0)*int(self.normalizer)
                    if self.C_channels == 1:
                        plot_msg = np.dstack((plot_msg, plot_msg, plot_msg))
                    self.plot_pub.publish(CvBridge().cv2_to_imgmsg(plot_msg.astype(np.uint8),'rgb8'))


if __name__ == "__main__":
    m = PredictiveCoding(module_name='predidctive_coding', steps=1)
    rospy.spin()