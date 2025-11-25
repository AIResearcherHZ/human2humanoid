import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from phc.utils import torch_utils
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)

import joblib
import torch
import torch.nn.functional as F
import math
from phc.utils.pytorch3d_transforms import axis_angle_to_matrix
from torch.autograd import Variable
from scipy.ndimage import gaussian_filter1d
from tqdm.notebook import tqdm
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPLH_MUJOCO_NAMES
from phc.utils.torch_Taks_T1_humanoid_batch import Humanoid_Batch_Taks_T1, TAKS_T1_ROTATION_AXIS

# Taks_T1 关节名称 (按照 MJCF 深度优先遍历顺序)
# 共 33 个 body (包括 pelvis)，32 个自由度
Taks_T1_joint_names = [
    'pelvis',  # 0
    # Left Leg (1-6)
    'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link',
    'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link',
    # Right Leg (7-12)
    'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link',
    'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link',
    # Waist (13-15)
    'waist_yaw_link', 'waist_roll_link', 'torso_link',
    # Left Arm (16-22)
    'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link',
    'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_yaw_link', 'left_wrist_pitch_link',
    # Right Arm (23-29)
    'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link',
    'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_yaw_link', 'right_wrist_pitch_link',
    # Neck (30-32)
    'neck_yaw_link', 'neck_roll_link', 'neck_pitch_link'
]

Taks_T1_fk = Humanoid_Batch_Taks_T1(extend_head=True)  # load forward kinematics model

# 定义 Taks_T1 和 SMPL 关节的对应关系
# extend_hand=True 添加 left_hand_link(33), right_hand_link(34)
# extend_head=True 添加 head_link(35)
Taks_T1_joint_names_augment = Taks_T1_joint_names + ["left_hand_link", "right_hand_link", "head_link"]
Taks_T1_joint_pick = [
    'pelvis',
    'left_hip_yaw_link', 'left_knee_link', 'left_ankle_roll_link',
    'right_hip_yaw_link', 'right_knee_link', 'right_ankle_roll_link',
    'left_shoulder_roll_link', 'left_elbow_link', 'left_hand_link',
    'right_shoulder_roll_link', 'right_elbow_link', 'right_hand_link',
    'head_link'
]
smpl_joint_pick = [
    "Pelvis",
    "L_Hip", "L_Knee", "L_Ankle",
    "R_Hip", "R_Knee", "R_Ankle",
    "L_Shoulder", "L_Elbow", "L_Hand",
    "R_Shoulder", "R_Elbow", "R_Hand",
    "Head"
]
Taks_T1_joint_pick_idx = [Taks_T1_joint_names_augment.index(j) for j in Taks_T1_joint_pick]
smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

# 准备拟合变量
device = torch.device("cpu")
num_dof = 32  # Taks_T1 有 32 个自由度

dof_pos = torch.zeros((1, num_dof))
pose_aa_Taks_T1 = torch.cat([
    torch.zeros((1, 1, 3)),
    TAKS_T1_ROTATION_AXIS * dof_pos[..., None],
    torch.zeros((1, 2, 3))
], axis=1)

root_trans = torch.zeros((1, 1, 3))

# 准备 SMPL 默认姿态
pose_aa_stand = np.zeros((1, 72))
rotvec = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).as_rotvec()
pose_aa_stand[:, :3] = rotvec
pose_aa_stand = pose_aa_stand.reshape(-1, 24, 3)
pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('L_Shoulder')] = sRot.from_euler("xyz", [0, 0, -np.pi/2], degrees=False).as_rotvec()
pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('R_Shoulder')] = sRot.from_euler("xyz", [0, 0, np.pi/2], degrees=False).as_rotvec()
pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('L_Elbow')] = sRot.from_euler("xyz", [0, -np.pi/2, 0], degrees=False).as_rotvec()
pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('R_Elbow')] = sRot.from_euler("xyz", [0, np.pi/2, 0], degrees=False).as_rotvec()
pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 72))

smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")

# 形状拟合
trans = torch.zeros([1, 3])
beta = torch.zeros([1, 10])
verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, beta, trans)
offset = joints[:, 0] - trans
root_trans_offset = trans + offset

fk_return = Taks_T1_fk.fk_batch(pose_aa_Taks_T1[None, ], root_trans_offset[None, 0:1])

shape_new = Variable(torch.zeros([1, 10]).to(device), requires_grad=True)
scale = Variable(torch.ones([1]).to(device), requires_grad=True)
optimizer_shape = torch.optim.Adam([shape_new, scale], lr=0.1)

for iteration in range(1000):
    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, shape_new, trans[0:1])
    root_pos = joints[:, 0]
    joints = (joints - joints[:, 0]) * scale + root_pos
    diff = fk_return.global_translation_extend[:, :, Taks_T1_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
    loss_g = diff.norm(dim=-1).mean()
    loss = loss_g
    if iteration % 100 == 0:
        print(iteration, loss.item() * 1000)

    optimizer_shape.zero_grad()
    loss.backward()
    optimizer_shape.step()

os.makedirs("data/Taks_T1", exist_ok=True)
joblib.dump((shape_new.detach(), scale), "data/Taks_T1/shape_optimized_v1.pkl")
print(f"shape fitted and saved to data/Taks_T1/shape_optimized_v1.pkl")
