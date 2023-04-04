import os

from matplotlib import pyplot as plt


def plot_pose(pose, cur_frame, prefix):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]
    ax.cla()
    num_joint = pose.shape[0] // 3
    for i, p in enumerate(parents):
        if i > 0:
            ax.plot([pose[i, 0], pose[p, 0]],\
                    [pose[i, 2], pose[p, 2]],\
                    [pose[i, 1], pose[p, 1]], c='r')
            ax.plot([pose[i+num_joint, 0], pose[p+num_joint, 0]],\
                    [pose[i+num_joint, 2], pose[p+num_joint, 2]],\
                    [pose[i+num_joint, 1], pose[p+num_joint, 1]], c='b')
            ax.plot([pose[i+num_joint*2, 0], pose[p+num_joint*2, 0]],\
                    [pose[i+num_joint*2, 2], pose[p+num_joint*2, 2]],\
                    [pose[i+num_joint*2, 1], pose[p+num_joint*2, 1]], c='g')
    ax.scatter(pose[:num_joint, 0], pose[:num_joint, 2], pose[:num_joint, 1],c='b')
    ax.scatter(pose[num_joint:num_joint*2, 0], pose[num_joint:num_joint*2, 2], pose[num_joint:num_joint*2, 1],c='b')
    ax.scatter(pose[num_joint*2:num_joint*3, 0], pose[num_joint*2:num_joint*3, 2], pose[num_joint*2:num_joint*3, 1],c='g')
    xmin = np.min(pose[:, 0])
    ymin = np.min(pose[:, 2])
    zmin = np.min(pose[:, 1])
    xmax = np.max(pose[:, 0])
    ymax = np.max(pose[:, 2])
    zmax = np.max(pose[:, 1])
    scale = np.max([xmax - xmin, ymax - ymin, zmax - zmin])
    xmid = (xmax + xmin) // 2
    ymid = (ymax + ymin) // 2
    zmid = (zmax + zmin) // 2
    ax.set_xlim(xmid - scale // 2, xmid + scale // 2)
    ax.set_ylim(ymid - scale // 2, ymid + scale // 2)
    ax.set_zlim(zmid - scale // 2, zmid + scale // 2)

    plt.draw()
    plt.savefig(f"{prefix}_{cur_frame:02}.png", dpi=200, bbox_inches='tight')
    plt.close()


def save_video(frames_loc, filepath):
    fps = 30

    frames = [os.path.join(frames_loc, img) for img in os.listdir(frames_loc) if
              img.endswith(".png") and img.startswith("pred")]
    frames.sort()
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(filepath)


with torch.no_grad():
    # State inputs
    local_q = torch.tensor(sequence['local_q'], dtype=torch.float32).unsqueeze(0).to(device)
    root_v = torch.tensor(sequence['root_v'], dtype=torch.float32).unsqueeze(0).to(device)
    contact = torch.tensor(sequence['contact'], dtype=torch.float32).unsqueeze(0).to(device)

    # Offset inputs
    root_p_offset = torch.tensor(sequence['root_p_offset'], dtype=torch.float32).unsqueeze(0).to(device)
    local_q_offset = torch.tensor(sequence['local_q_offset'], dtype=torch.float32).unsqueeze(0).to(device)
    local_q_offset = local_q_offset.view(local_q_offset.size(0), -1)

    # Target inputs
    target = torch.tensor(sequence['target'], dtype=torch.float32).unsqueeze(0).to(device)
    target = target.view(target.size(0), -1)

    # Root position
    root_p = torch.tensor(sequence['root_p'], dtype=torch.float32).unsqueeze(0).to(device)

    # X
    X = torch.tensor(sequence['X'], dtype=torch.float32).unsqueeze(0).to(device)

    lstm.init_hidden(local_q.size(0))

    root_pred = None
    local_q_pred = None
    contact_pred = None
    root_v_pred = None

    for t in tqdm(range(sequence_length - 1)):
        if t == 0:
            root_p_t = root_p[:, t]
            local_q_t = local_q[:, t]
            local_q_t = local_q_t.view(local_q_t.size(0), -1)
            contact_t = contact[:, t]
            root_v_t = root_v[:, t]
        else:
            root_p_t = root_pred[0]
            local_q_t = local_q_pred[0]
            contact_t = contact_pred[0]
            root_v_t = root_v_pred[0]

        state_input = torch.cat([local_q_t, root_v_t, contact_t], -1)

        root_p_offset_t = root_p_offset - root_p_t
        local_q_offset_t = local_q_offset - local_q_t
        offset_input = torch.cat([root_p_offset_t, local_q_offset_t], -1)

        target_input = target

        h_state = state_encoder(state_input)
        h_offset = offset_encoder(offset_input)
        h_target = target_encoder(target_input)

        tta = sequence_length - t - 2

        h_state += ztta[tta]
        h_offset += ztta[tta]
        h_target += ztta[tta]

        #         if tta < 5:
        #             lambda_target = 0.0
        #         elif tta >= 5 and tta < 30:
        #             lambda_target = (tta - 5) / 25.0
        #         else:
        #             lambda_target = 1.0
        #         h_offset += 0.5 * lambda_target * torch.FloatTensor(h_offset.size()).normal_().to(device)
        #         h_target += 0.5 * lambda_target * torch.FloatTensor(h_target.size()).normal_().to(device)

        h_in = torch.cat([h_state, h_offset, h_target], -1).unsqueeze(0)
        h_out = lstm(h_in)

        h_pred, contact_pred = decoder(h_out)
        local_q_v_pred = h_pred[:, :, :88]
        local_q_pred = local_q_v_pred + local_q_t

        local_q_pred_ = local_q_pred.view(local_q_pred.size(0), local_q_pred.size(1), -1, 4)
        local_q_pred_ = local_q_pred_ / torch.norm(local_q_pred_, dim=-1, keepdim=True)

        root_v_pred = h_pred[:, :, 88:]
        root_pred = root_v_pred + root_p_t

        pos_pred = skeleton.forward_kinematics(local_q_pred_, root_pred)

        local_q_next = local_q[:, t + 1]
        local_q_next = local_q_next.view(local_q_next.size(0), -1)

        # Saving images
        plot_pose(np.concatenate([X[0, 0].view(22, 3).detach().cpu().numpy(), \
                                  pos_pred[0, 0].view(22, 3).detach().cpu().numpy(), \
                                  X[0, -1].view(22, 3).detach().cpu().numpy()], 0), \
                  t, './results/temp/pred')
#         plot_pose(np.concatenate([X[0, 0].view(22, 3).detach().cpu().numpy(),\
#                                 X[0, t+1].view(22, 3).detach().cpu().numpy(),\
#                                 X[0, -1].view(22, 3).detach().cpu().numpy()], 0),\
#                                 t, './results/temp/gt')

save_video("./results/temp/", "./results/sub1234_2sec_trial_4_epoch_200_01.mp4")