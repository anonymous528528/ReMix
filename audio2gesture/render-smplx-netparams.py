import os, sys
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import numpy as np
import cv2
import os.path as osp
import pyrender
import trimesh
import json
import pickle
import smplx
import torch
from human_body_prior.tools.model_loader import load_vposer
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import torch.nn.functional as F
from camera import PerspectiveCamera

device = torch.device('cuda')
dtype = torch.float32
camera = PerspectiveCamera(focal_length_x = 5000.0,
                        focal_length_y = 5000.0,
                        dtype=dtype,
                        batch_size=1)
camera = camera.to(device=device)
with torch.no_grad():
    camera.center[:] = torch.tensor([1080, 1080], dtype=dtype) * 0.5



def render_mesh(mesh_trimesh, camera_center, camera_transl, focal_length, img_width, img_height):
    rot = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0])
    mesh_trimesh.apply_transform(rot)    ##x180
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(1.0, 1.0, 0.9, 1.0))

    vertex_colors = np.loadtxt('smplx_verts_colors.txt')
    mesh_new = trimesh.Trimesh(vertices=mesh_trimesh.vertices, faces=mesh_trimesh.faces, vertex_colors=vertex_colors)
    mesh_new.vertex_colors = vertex_colors
    print("mesh visual kind: %s" % mesh_new.visual.kind)


    mesh = pyrender.Mesh.from_trimesh(mesh_new, smooth=False, wireframe=False)

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                           ambient_light=(0.3, 0.3, 0.3))

    scene.add(mesh, 'mesh')

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = camera_transl[:]
    camera = pyrender.camera.IntrinsicsCamera(
        fx=focal_length, fy=focal_length,
        cx=camera_center[0], cy=camera_center[1])
    scene.add(camera, pose=camera_pose)

    light = pyrender.light.DirectionalLight()

    scene.add(light)

    r = pyrender.OffscreenRenderer(viewport_width=img_width,
                                   viewport_height=img_height,
                                   point_size=1.0)

    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0

    output_img = color[:, :, 0:3]
    output_img = (output_img * 255).astype(np.uint8)
    
    return output_img

out_path = '../examples/rendering'
if not os.path.exists(out_path):
    os.mkdir(out_path)
######################load smplx params npz
params = np.load(open('traindata/params/A/A.npz', 'rb'))
camera_translation_npz = params['camera_translation']
betas_npz = params['betas']
global_orient_npz = params['global_orient']
left_hand_pose_npz = params['left_hand_pose']
right_hand_pose_npz = params['right_hand_pose']

jaw_pose_npz = params['jaw_pose']
leye_pose_npz = params['leye_pose']
reye_pose_npz = params['reye_pose']
expression_npz = params['expression']
body_pose_npz = params['body_pose']
body_pose_embedding_npz = params['body_pose_embedding']


stdcamera = np.std(camera_translation_npz, axis=0)
meancamera = np.mean(camera_translation_npz,axis=0)


stdbody_pose = np.std(body_pose_npz, axis=0)
meanbody_pose = np.mean(body_pose_npz,axis=0)

stdbody_pose_embedding = np.std(body_pose_embedding_npz, axis=0)
meanbody_pose_embedding = np.mean(body_pose_embedding_npz,axis=0)

stdjaw_pose = np.std(jaw_pose_npz, axis=0)
meanjaw_pose = np.mean(jaw_pose_npz,axis=0)

stdleft_hand_pose = np.std(left_hand_pose_npz, axis=0)
meanleft_hand_pose = np.mean(left_hand_pose_npz,axis=0)

stdright_hand_pose = np.std(right_hand_pose_npz, axis=0)
meanright_hand_pose = np.mean(right_hand_pose_npz,axis=0)

stdexpression = np.std(expression_npz, axis=0)
meanexpression = np.mean(expression_npz,axis=0)


netparams = np.load(open('../examples/test-result/A_test.npz', 'rb'))

netparams = netparams['face']

netparams[:,:] = savgol_filter(netparams[:,:], 11, 5, axis=0)

print('netparams.shape:', netparams.shape)

#############vposer
dtype = torch.float32
pose_embedding = torch.zeros([1, 32], dtype=dtype, device=device,
                                     requires_grad=True)

body_pose = torch.zeros([1, 63], dtype=dtype, device=device,
                                     requires_grad=False)

vposer_ckpt = osp.expandvars('vposer/vposer_v1_0')
vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
vposer = vposer.cuda()
vposer.eval()

########smplx  mesh

model_params = {'model_path': 'models', 'create_global_orient': True, 'create_body_pose': False, 'create_betas': True, 'create_left_hand_pose': True, 'create_right_hand_pose': True, 'create_expression': True, 'create_jaw_pose': True, 'create_leye_pose': True, 'create_reye_pose': True, 'create_transl': False, 'dtype': torch.float32, 'data_folder': '../data', 'config': 'cfg_files/fit_smplx.yaml', 'loss_type': 'smplify', 'interactive': True, 'save_meshes': True, 'visualize': True, 'degrees': [0, 90, 180, 270], 'use_cuda': True, 'dataset': 'openpose', 'joints_to_ign': [1, 9, 12], 'keyp_folder': 'keypoints', 'summary_folder': 'summaries', 'float_dtype': 'float32', 'model_type': 'smplx', 'camera_type': 'persp', 'optim_jaw': True, 'optim_hands': True, 'optim_expression': True, 'optim_shape': True, 'model_folder': '../smplx/models', 'use_joints_conf': True, 'batch_size': 1, 'num_gaussians': 8, 'use_pca': True, 'num_pca_comps': 12, 'flat_hand_mean': False, 'body_prior_type': 'l2', 'left_hand_prior_type': 'l2', 'right_hand_prior_type': 'l2', 'jaw_prior_type': 'l2', 'use_vposer': True, 'vposer_ckpt': '../vposer/vposer_v1_0', 'init_joints_idxs': [9, 12, 2, 5], 'body_tri_idxs': [(5, 12), (2, 9)], 'prior_folder': 'priors', 'focal_length': 5000.0, 'rho': 100.0, 'interpenetration': False, 'penalize_outside': True, 'data_weights': [1, 1, 1, 1, 1], 'body_pose_prior_weights': [404.0, 404.0, 57.4, 4.78, 4.78], 'shape_weights': [100.0, 50.0, 10.0, 5.0, 5.0], 'expr_weights': [100.0, 50.0, 10.0, 5.0, 5.0], 'face_joints_weights': [0.0, 0.0, 0.0, 0.0, 2.0], 'hand_joints_weights': [0.0, 0.0, 0.0, 0.1, 2.0], 'jaw_pose_prior_weights': ['4.04e03,4.04e04,4.04e04', '4.04e03,4.04e04,4.04e04', '574,5740,5740', '47.8,478,478', '47.8,478,478'], 'hand_pose_prior_weights': [404.0, 404.0, 57.4, 4.78, 4.78], 'coll_loss_weights': [0.0, 0.0, 0.0, 0.01, 1.0], 'depth_loss_weight': 100.0, 'df_cone_height': 0.0001, 'max_collisions': 128, 'point2plane': False, 'part_segm_fn': 'smplx_parts_segm.pkl', 'ign_part_pairs': ['9,16', '9,17', '6,16', '6,17', '1,2', '12,22'], 'use_hands': True, 'use_face': True, 'use_face_contour': False, 'side_view_thsh': 25, 'optim_type': 'lbfgsls', 'lr': 1.0, 'gtol': 1e-09, 'ftol': 1e-09, 'maxiters': 30}
body_model = smplx.create(gender='male', **model_params).to(device=device)

for fid in range(1,netparams.shape[0]):
    print(fid)
    strnum = '%05d' % fid
    
    ############# add params to smplx model
    body_model.betas[0,:] = torch.tensor(betas_npz[0])
    body_model.global_orient[0,:] = torch.tensor(global_orient_npz[0])
    

    w1 = 32

    body_model.jaw_pose[0,:] = torch.tensor(netparams[fid-1,w1+37:w1+40])
    body_model.left_hand_pose[0,:] = torch.tensor(netparams[fid-1,w1+3:w1+15])
    body_model.right_hand_pose[0,:] = torch.tensor(netparams[fid-1,w1+15:w1+27])
    body_model.expression[0,:] = torch.tensor(netparams[fid-1,w1+27:w1+37])

    pose_embedding[0,:] = torch.tensor(netparams[fid-1,:w1])
    body_pose = vposer.decode(
        pose_embedding,
        output_type='aa').view(1, -1)
    model_output = body_model(return_verts=True,body_pose=body_pose)


    trans = camera_translation_npz[fid-1].copy()
    camera_translation_fid = torch.tensor(trans,dtype=dtype)
    with torch.no_grad():
        camera.translation[:] = camera_translation_fid.view_as(camera.translation)

    vertices = model_output.vertices.detach().cpu().numpy().squeeze()
    trimesh_obj = trimesh.Trimesh(vertices, body_model.faces, process=False)


    trans0 = camera_translation_npz[0].copy()
    trans0[0] *= -1.0
    trans = netparams[fid-1,w1:w1+3]*(stdcamera/2+0.01) + meancamera

    trans[0] *= -1.0
    h = 1080
    w = 1080

    output_img = render_mesh(trimesh_obj, [w/2,h/2], trans0, 5000.0, w, h)
    output_img_cv = np.array(output_img)
    output_img_cv = cv2.cvtColor(output_img_cv, cv2.COLOR_RGB2BGR)
    
    final_image = output_img_cv

    cv2.imwrite(os.path.join(out_path,strnum+'.png'), final_image)  



