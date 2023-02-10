import torch
from smplx import build_layer
from smplx.lbs import blend_shapes, vertices2joints

def batch_rodrigues(rot_vecs, epsilon: float = 1e-8):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''
    assert len(rot_vecs.shape) == 2, (
        f'Expects an array of size Bx3, but received {rot_vecs.shape}')

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device
    dtype = rot_vecs.dtype

    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True, p=2)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def init_smpl(model_folder, model_type, gender, num_betas, device='cuda'):
    if device == 'cuda':
        smpl_model = build_layer(
            model_folder, model_type = model_type,
            gender = gender, num_betas = num_betas
        ).cuda()
    elif device == 'cpu':
        smpl_model = build_layer(
            model_folder, model_type = model_type,
            gender = gender, num_betas = num_betas
        )
    return smpl_model

def get_J(beta, smpl_model):
    beta = beta.reshape([1, 10]).cuda()
    v_shaped = smpl_model.v_template + blend_shapes(beta, smpl_model.shapedirs)
    J = vertices2joints(smpl_model.J_regressor, v_shaped)
    return J.reshape(-1, 3).detach().cpu().numpy()

def get_J_batch_cpu(beta, smpl_model):
    beta = beta.reshape([-1, 10])
    v_shaped = smpl_model.v_template + blend_shapes(beta, smpl_model.shapedirs)
    J = vertices2joints(smpl_model.J_regressor, v_shaped)
    return J.reshape(-1, 3).detach().numpy()

def get_shape_pose(beta, pose, smpl_model, bs):
    beta = beta.reshape([bs, 10]).cuda()
    pose_rot = batch_rodrigues(pose.reshape(-1, 3)).reshape(bs, 24, 3, 3)
    so = smpl_model(betas = beta, body_pose = pose_rot[:, 1:], global_orient = pose_rot[:, 0, :, :].view(bs, 1, 3, 3))
    joints = so['joints'].reshape()
    return joints
