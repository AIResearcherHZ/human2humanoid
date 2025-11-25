import torch 
import numpy as np
import phc.utils.rotation_conversions as tRot
import xml.etree.ElementTree as ETree
from easydict import EasyDict
import scipy.ndimage.filters as filters
import smpl_sim.poselib.core.rotation3d as pRot

# Taks_T1 关节旋转轴定义 (32个自由度)
# 按照 MJCF 深度优先遍历顺序，与 from_mjcf 解析顺序一致
TAKS_T1_ROTATION_AXIS = torch.tensor([[
    # Left Leg (6 DOF) - body index 1-6
    [0, 1, 0],  # 1. left_hip_pitch_joint (axis="0 1 0")
    [1, 0, 0],  # 2. left_hip_roll_joint (axis="1 0 0")
    [0, 0, 1],  # 3. left_hip_yaw_joint (axis="0 0 1")
    [0, 1, 0],  # 4. left_knee_joint (axis="0 1 0")
    [0, 1, 0],  # 5. left_ankle_pitch_joint (axis="0 1 0")
    [1, 0, 0],  # 6. left_ankle_roll_joint (axis="1 0 0")

    # Right Leg (6 DOF) - body index 7-12
    [0, 1, 0],  # 7. right_hip_pitch_joint (axis="0 1 0")
    [1, 0, 0],  # 8. right_hip_roll_joint (axis="1 0 0")
    [0, 0, 1],  # 9. right_hip_yaw_joint (axis="0 0 1")
    [0, 1, 0],  # 10. right_knee_joint (axis="0 1 0")
    [0, 1, 0],  # 11. right_ankle_pitch_joint (axis="0 1 0")
    [1, 0, 0],  # 12. right_ankle_roll_joint (axis="1 0 0")

    # Waist (3 DOF) - body index 13-15
    [0, 0, 1],  # 13. waist_yaw_joint (axis="0 0 1")
    [1, 0, 0],  # 14. waist_roll_joint (axis="1 0 0")
    [0, 1, 0],  # 15. waist_pitch_joint (axis="0 1 0")

    # Left Arm (7 DOF) - body index 16-22
    [0, 1, 0],  # 16. left_shoulder_pitch_joint (axis="0 1 0")
    [1, 0, 0],  # 17. left_shoulder_roll_joint (axis="1 0 0")
    [0, 0, 1],  # 18. left_shoulder_yaw_joint (axis="0 0 1")
    [0, 1, 0],  # 19. left_elbow_joint (axis="0 1 0")
    [1, 0, 0],  # 20. left_wrist_roll_joint (axis="1 0 0")
    [0, 0, 1],  # 21. left_wrist_yaw_joint (axis="0 0 1")
    [0, 1, 0],  # 22. left_wrist_pitch_joint (axis="0 1 0")

    # Right Arm (7 DOF) - body index 23-29
    [0, 1, 0],  # 23. right_shoulder_pitch_joint (axis="0 1 0")
    [1, 0, 0],  # 24. right_shoulder_roll_joint (axis="1 0 0")
    [0, 0, 1],  # 25. right_shoulder_yaw_joint (axis="0 0 1")
    [0, 1, 0],  # 26. right_elbow_joint (axis="0 1 0")
    [1, 0, 0],  # 27. right_wrist_roll_joint (axis="1 0 0")
    [0, 0, 1],  # 28. right_wrist_yaw_joint (axis="0 0 1")
    [0, 1, 0],  # 29. right_wrist_pitch_joint (axis="0 1 0")

    # Neck (3 DOF) - body index 30-32
    [0, 0, 1],  # 30. neck_yaw_joint (axis="0 0 1")
    [1, 0, 0],  # 31. neck_roll_joint (axis="1 0 0")
    [0, 1, 0],  # 32. neck_pitch_joint (axis="0 1 0")
]])


class Humanoid_Batch_Taks_T1:

    def __init__(self, mjcf_file=f"resources/robots/Taks_T1/Taks_T1.xml", extend_hand=True, extend_head=False, device=torch.device("cpu")):
        self.mjcf_data = mjcf_data = self.from_mjcf(mjcf_file)
        self.extend_hand = extend_hand
        self.extend_head = extend_head
        
        if extend_hand:
            self.model_names = mjcf_data['node_names'] + ["left_hand_link", "right_hand_link"]
            # left_wrist_pitch_link 和 right_wrist_pitch_link 的索引
            left_wrist_idx = mjcf_data['node_names'].index('left_wrist_pitch_link')
            right_wrist_idx = mjcf_data['node_names'].index('right_wrist_pitch_link')
            self._parents = torch.cat((mjcf_data['parent_indices'], torch.tensor([left_wrist_idx, right_wrist_idx]))).to(device)
            arm_length = 0.15  # Taks_T1 手部延伸长度
            self._offsets = torch.cat((mjcf_data['local_translation'], torch.tensor([[arm_length, 0, 0], [arm_length, 0, 0]])), dim=0)[None, ].to(device)
            self._local_rotation = torch.cat((mjcf_data['local_rotation'], torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])), dim=0)[None, ].to(device)
            self._remove_idx = 2
        else:
            self._parents = mjcf_data['parent_indices']
            self.model_names = mjcf_data['node_names']
            self._offsets = mjcf_data['local_translation'][None, ].to(device)
            self._local_rotation = mjcf_data['local_rotation'][None, ].to(device)
            
        if extend_head:
            self._remove_idx = 3
            self.model_names = self.model_names + ["head_link"]
            self._parents = torch.cat((self._parents, torch.tensor([0]).to(device))).to(device)
            head_length = 0.75
            self._offsets = torch.cat((self._offsets, torch.tensor([[[0, 0, head_length]]]).to(device)), dim=1).to(device)
            self._local_rotation = torch.cat((self._local_rotation, torch.tensor([[[1, 0, 0, 0]]]).to(device)), dim=1).to(device)
            
        self.joints_range = mjcf_data['joints_range'].to(device)
        self._local_rotation_mat = tRot.quaternion_to_matrix(self._local_rotation).float()
        
    def from_mjcf(self, path):
        tree = ETree.parse(path)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        if xml_world_body is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
        xml_body_root = xml_world_body.find("body")
        if xml_body_root is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
            
        node_names = []
        parent_indices = []
        local_translation = []
        local_rotation = []
        joints_range = []

        def _add_xml_node(xml_node, parent_index, node_index):
            node_name = xml_node.attrib.get("name")
            pos = np.fromstring(xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" ")
            quat = np.fromstring(xml_node.attrib.get("quat", "1 0 0 0"), dtype=float, sep=" ")
            node_names.append(node_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            local_rotation.append(quat)
            curr_index = node_index
            node_index += 1
            all_joints = xml_node.findall("joint")
            for joint in all_joints:
                if not joint.attrib.get("range") is None: 
                    joints_range.append(np.fromstring(joint.attrib.get("range"), dtype=float, sep=" "))
            
            for next_node in xml_node.findall("body"):
                node_index = _add_xml_node(next_node, curr_index, node_index)
            return node_index
        
        _add_xml_node(xml_body_root, -1, 0)
        return {
            "node_names": node_names,
            "parent_indices": torch.from_numpy(np.array(parent_indices, dtype=np.int32)),
            "local_translation": torch.from_numpy(np.array(local_translation, dtype=np.float32)),
            "local_rotation": torch.from_numpy(np.array(local_rotation, dtype=np.float32)),
            "joints_range": torch.from_numpy(np.array(joints_range))
        }

        
    def fk_batch(self, pose, trans, convert_to_mat=True, return_full=False, dt=1/30):
        device, dtype = pose.device, pose.dtype
        pose_input = pose.clone()
        B, seq_len = pose.shape[:2]
        pose = pose[..., :len(self._parents), :]
        
        if self.extend_hand and self.extend_head and pose.shape[-2] < len(self._parents):
            extra_joints = len(self._parents) - pose.shape[-2]
            pose = torch.cat([pose, torch.zeros(B, seq_len, extra_joints, 3).to(device).type(dtype)], dim=-2)

        if convert_to_mat:
            pose_quat = tRot.axis_angle_to_quaternion(pose)
            pose_mat = tRot.quaternion_to_matrix(pose_quat)
        else:
            pose_mat = pose
        if pose_mat.shape != 5:
            pose_mat = pose_mat.reshape(B, seq_len, -1, 3, 3)
        J = pose_mat.shape[2] - 1
        
        wbody_pos, wbody_mat = self.forward_kinematics_batch(pose_mat[:, :, 1:], pose_mat[:, :, 0:1], trans)
        
        return_dict = EasyDict()
        
        wbody_rot = tRot.wxyz_to_xyzw(tRot.matrix_to_quaternion(wbody_mat))
        if self.extend_hand:
            if return_full:
                return_dict.global_velocity_extend = self._compute_velocity(wbody_pos, dt) 
                return_dict.global_angular_velocity_extend = self._compute_angular_velocity(wbody_rot, dt)
                
            return_dict.global_translation_extend = wbody_pos.clone()
            return_dict.global_rotation_mat_extend = wbody_mat.clone()
            return_dict.global_rotation_extend = wbody_rot
            
            wbody_pos = wbody_pos[..., :-self._remove_idx, :]
            wbody_mat = wbody_mat[..., :-self._remove_idx, :, :]
            wbody_rot = wbody_rot[..., :-self._remove_idx, :]
        
        return_dict.global_translation = wbody_pos
        return_dict.global_rotation_mat = wbody_mat
        return_dict.global_rotation = wbody_rot
            
        if return_full:
            rigidbody_linear_velocity = self._compute_velocity(wbody_pos, dt)
            rigidbody_angular_velocity = self._compute_angular_velocity(wbody_rot, dt)
            return_dict.local_rotation = tRot.wxyz_to_xyzw(pose_quat)
            return_dict.global_root_velocity = rigidbody_linear_velocity[..., 0, :]
            return_dict.global_root_angular_velocity = rigidbody_angular_velocity[..., 0, :]
            return_dict.global_angular_velocity = rigidbody_angular_velocity
            return_dict.global_velocity = rigidbody_linear_velocity
            
            if self.extend_hand or self.extend_head:
                return_dict.dof_pos = pose.sum(dim=-1)[..., 1:][..., :-self._remove_idx]
            else:
                return_dict.dof_pos = pose.sum(dim=-1)[..., 1:]
            
            dof_vel = ((return_dict.dof_pos[:, 1:] - return_dict.dof_pos[:, :-1]) / dt)
            return_dict.dof_vels = torch.cat([dof_vel, dof_vel[:, -2:-1]], dim=1)
            return_dict.fps = int(1/dt)
        
        return return_dict
    

    def forward_kinematics_batch(self, rotations, root_rotations, root_positions):
        device, dtype = root_rotations.device, root_rotations.dtype
        B, seq_len = rotations.size()[0:2]
        J = self._offsets.shape[1]
        positions_world = []
        rotations_world = []

        expanded_offsets = (self._offsets[:, None].expand(B, seq_len, J, 3).to(device).type(dtype))

        for i in range(J):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(root_rotations)
            else:
                jpos = (torch.matmul(rotations_world[self._parents[i]][:, :, 0], expanded_offsets[:, :, i, :, None]).squeeze(-1) + positions_world[self._parents[i]])
                rot_mat = torch.matmul(rotations_world[self._parents[i]], torch.matmul(self._local_rotation_mat[:, (i):(i + 1)], rotations[:, :, (i - 1):i, :]))
                
                positions_world.append(jpos)
                rotations_world.append(rot_mat)
        
        positions_world = torch.stack(positions_world, dim=2)
        rotations_world = torch.cat(rotations_world, dim=2)
        return positions_world, rotations_world
    
    @staticmethod
    def _compute_velocity(p, time_delta, guassian_filter=True):
        velocity = np.gradient(p.numpy(), axis=-3) / time_delta
        if guassian_filter:
            velocity = torch.from_numpy(filters.gaussian_filter1d(velocity, 2, axis=-3, mode="nearest")).to(p)
        else:
            velocity = torch.from_numpy(velocity).to(p)
        
        return velocity
    
    @staticmethod
    def _compute_angular_velocity(r, time_delta: float, guassian_filter=True):
        diff_quat_data = pRot.quat_identity_like(r).to(r)
        diff_quat_data[..., :-1, :, :] = pRot.quat_mul_norm(r[..., 1:, :, :], pRot.quat_inverse(r[..., :-1, :, :]))
        diff_angle, diff_axis = pRot.quat_angle_axis(diff_quat_data)
        angular_velocity = diff_axis * diff_angle.unsqueeze(-1) / time_delta
        if guassian_filter:
            angular_velocity = torch.from_numpy(filters.gaussian_filter1d(angular_velocity.numpy(), 2, axis=-3, mode="nearest"),)
        return angular_velocity
