# Imported transfer function that trains a predictive coding network live, and make predictions
from sensor_msgs.msg import Image
from std_msgs.msg    import Float32MultiArray
@nrp.MapRobotSubscriber('camera',         Topic('/camera/image_raw',     Image))
@nrp.MapRobotPublisher( 'plot_topic',     Topic('/pred_plot',            Image))
@nrp.MapRobotPublisher( 'latent_topic',   Topic('/latent',   Float32MultiArray))
@nrp.MapRobotPublisher( 'pred_pos_topic', Topic('/pred_pos', Float32MultiArray))
@nrp.MapVariable(       'pred_msg',       initial_value=None)
@nrp.MapVariable(       'model',          initial_value=None)
@nrp.MapVariable(       'model_path',     initial_value=None)
@nrp.MapVariable(       'model_inputs',   initial_value=None)
@nrp.MapVariable(       'optimizer',      initial_value=None)
@nrp.MapVariable(       'scheduler',      initial_value=None)
@nrp.MapVariable(       'run_step',       initial_value=0   )
@nrp.Robot2Neuron()
def img_to_pred(t, camera, plot_topic, latent_topic, pred_pos_topic, pred_msg,
    model, model_path, model_inputs, optimizer, scheduler, run_step):

    # Imports
    import os
    import torch
    import torch.nn.functional as F
    import numpy as np
    from prednet      import PredNet
    from cv_bridge    import CvBridge
    from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension
    from specs        import localize_target, complete_target_positions, mark_target, exp_dir

    # Image and model parameters
    underSmpl      = 5      # Avoiding too sharp time resolution (no change between frames)
    nt             = 15     # Number of "past" frames given to the network
    t_extrap       = 5      # After this frame, input is not used for future predictions
    n_feat         = 1      # Factor for number of features used in the network
    max_pix_value  = 1.0
    normalizer     = 255.0/max_pix_value
    C_channels     = 3      # 1 or 3 (color channels)
    A_channels     = (C_channels, n_feat*4, n_feat*8, n_feat*16)
    R_channels     = (C_channels, n_feat*4, n_feat*8, n_feat*16)
    scale          = 4      # 2 or 4 (how much layers down/upsample images)
    pad            = 8 if scale == 4 else 0  # For up/downsampling to work
    model_name     = 'model' + str(n_feat)+'.pt'
    new_model_path = os.getcwd() + '/resources/' + model_name
    trained_w_path = exp_dir + model_name    # exp_dir computed in specs.py 
    device         = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # Training parameters
    use_new_w      = False  # If True, do not use weights that are saved in new_model_path
    use_trained_w  = True   # If above is False, use trained_w_path as model weights
    do_train       = False  # Train with present frames if True, predicts future if False
    initial_lr     = 1e-4   # Then, the learning rate is scheduled with cosine annealing
    epoch_loop     = 100    # Every epoch_loop, a prediction is made, to monitor progress
    n_batches      = 1      # For now, not usable (could roll images for multiple batches)
    
    # Check that the simulation frame is far enough
    if camera.value is not None and int(t*50) % underSmpl == 0:

        # Collect input image and initialize the network input
        cam_img = CvBridge().imgmsg_to_cv2(camera.value, 'rgb8')/normalizer
        if C_channels == 3:  # Below I messed up, it should be (2,0,1) but the model is already trained.
            cam_img = torch.tensor(cam_img, device=device).permute(2,1,0)  # --> channels last
        if C_channels == 1:
            cam_img = cam_img[:,:,1]  # .mean(axis=2)
            cam_img = torch.tensor(cam_img, device=device).unsqueeze(dim=2).permute(2,1,0)
        img_shp = cam_img.shape
        cam_img = F.pad(cam_img, (pad, pad), 'constant', 0.0)  # width may need to be 256
        if model_inputs.value is None:
            model_inputs.value = torch.zeros((1,nt)+cam_img.shape, device=device)

        # Update the model or the mode, if needed
        run_step.value = run_step.value + 1
        if new_model_path != model_path.value:

            # Update the model path if new or changed and reset prediction plot
            model_path.value = new_model_path
            pred_msg.value   = torch.ones(img_shp[0], img_shp[1]*(nt-t_extrap), img_shp[2]+10)*64.0

            # Load or reload the model
            model.value = PredNet(R_channels, A_channels, device=device, t_extrap=t_extrap, scale=scale)
            if device == 'cuda': model.value = model.value.to('cuda')
            if run_step.value == 1:
                try:
                    if use_new_w:
                        a = 1./0.
                    if use_trained_w:
                        model.value.load_state_dict(torch.load(trained_w_path, map_location=torch.device(device)))
                        clientLogger.info('Model initialized with pre-trained weights.')
                    else:
                        model.value.load_state_dict(torch.load(model_path.value))
                        clientLogger.info('Learning weights loaded in the model.')
                except:
                    clientLogger.info('No existing weight file found. Model initialized randomly.')
            
        # Initialize some variables needed for training
        time_loss_w = [1.0/(nt-1) if s > 0 else 0.0 for s in range(nt)]
        if t_extrap < nt:
            time_loss_w = [w if n < t_extrap else 2.0*w for n, w in enumerate(time_loss_w)]

        if None in [optimizer.value, scheduler.value]:
            optimizer.value = torch.optim.Adam(model.value.parameters(), lr=initial_lr)
            scheduler.value = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer.value, T_0=50)

        # Save the model at each epoch
        if run_step.value % epoch_loop == 1:
            torch.save(model.value.state_dict(), model_path.value)

        # Check that the model exists and initialize plot message
        if model.value is not None:

            # Feed network and train it or compute prediction
            model_inputs.value = model_inputs.value.roll(-1, dims=1)
            model_inputs.value[0,-1,:,:,:] = cam_img
            if run_step.value > nt:

                # Compute prediction along present frames and updates weights
                if do_train:

                    # Compute prediction loss for every frame
                    pred, latent = model.value(model_inputs.value, nt)
                    loss         = torch.tensor([0.0], device=device)
                    for s in range(nt):
                        error = (pred[s][0] - model_inputs.value[0][s])**2
                        loss += torch.sum(error)*time_loss_w[s]

                    # Backward pass and weight updates
                    optimizer.value.zero_grad()
                    loss.backward()
                    optimizer.value.step()
                    scheduler.value.step()

                # Predicts future frames without weight updates
                else:
                    with torch.no_grad():
                        pred, latent = model.value(model_inputs.value[:,-t_extrap:,:,:,:], nt)

                # Collect prediction frames
                displays = []
                targ_pos = []
                for s in range(nt-t_extrap):
                    disp = torch.detach(pred[t_extrap+s].clamp(0.0, 1.0)[0,:,:,pad:-pad]).cpu()
                    # disp = model_inputs.value[0,-(s+1),:,:,pad:-pad].cpu()  # for tests
                    targ_pos.append(localize_target(disp))
                    displays.append(disp)

                # Complete for missing target positions, highlight target and set the display message
                if 0 < np.sum([any([np.isnan(p) for p in pos]) for pos in targ_pos]) < len(targ_pos)-2:
                    targ_pos = complete_target_positions(targ_pos)
                for s, (disp, pos) in enumerate(zip(displays, targ_pos)):
                    pred_msg.value[:,s*img_shp[1]:(s+1)*img_shp[1],:img_shp[2]] = mark_target(disp, pos)

                # Print loss or prediction messages
                if do_train:
                    clientLogger.info('Epoch: %2i - step: %2i - error: %5.4f - lr: %5.4f' % \
                        (int(run_step.value/epoch_loop), run_step.value%epoch_loop, loss.item(), \
                         scheduler.value.get_lr()[0]))
                else:
                    clientLogger.info('Prediction for future target locations: ' + str(targ_pos))

                # Send latent state message (latent[0] to remove batch dimension)
                latent_msg = list(latent[0].cpu().numpy().flatten())
                layout_msg = MultiArrayLayout(dim=[MultiArrayDimension(size=d) for d in latent[0].shape])
                latent_topic.send_message(Float32MultiArray(layout=layout_msg, data=latent_msg))

                # Send predicted position according to the index of the frame that has to be reported
                pos_3d_msg = [[1.562-p[0]/156.274, -0.14-p[1]/152.691, 0.964+p[0]-p[0]] for p in targ_pos]
                pos_3d_msg = [p for pos in pos_3d_msg for p in pos]  # flatten the list
                layout_msg = MultiArrayLayout(dim=[MultiArrayDimension(size=d) for d in [len(targ_pos),3]])
                pred_pos_topic.send_message(Float32MultiArray(layout=layout_msg, data=pos_3d_msg))

            # Collect input frames
            inpt_msg = torch.zeros(img_shp[0], img_shp[1]*(nt-t_extrap), img_shp[2])
            for s in range(nt-t_extrap):
                inpt_msg[:,s*img_shp[1]:(s+1)*img_shp[1],:] = model_inputs.value[0,t_extrap+s,:,:,pad:-pad]

            # Build and send the display message
            plot_msg = torch.cat((pred_msg.value, inpt_msg), 2).numpy().transpose(2,1,0)*int(normalizer)
            if C_channels == 1:
                plot_msg = np.dstack((plot_msg, plot_msg, plot_msg))
            plot_topic  .send_message(CvBridge().cv2_to_imgmsg(plot_msg.astype(np.uint8),'rgb8'))
            