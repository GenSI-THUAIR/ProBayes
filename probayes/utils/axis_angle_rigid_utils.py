# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Any, Sequence, Callable, Optional

import numpy as np
import torch
from .so3_utils import rotquat_to_rotmat


class Rigid_AxisAngle:
    """
        A class representing a rigid transformation. Little more than a wrapper
        around two objects: a Rotation object and a [*, 3] translation
        Designed to behave approximately like a single torch tensor with the 
        shape of the shared batch dimensions of its component parts.
    """
    def __init__(self, 
        axis: torch.Tensor,
        angle: torch.Tensor,
        trans: torch.Tensor,
    ):
        """
            Args:
                rots: A [*, 3, 3] rotation tensor
                trans: A corresponding [*, 3] translation tensor
        """
        # (we need device, dtype, etc. from at least one input)

        batch_dims, dtype, device, requires_grad = None, None, None, None
        if(trans is not None):
            batch_dims = trans.shape[:-1]
            dtype = trans.dtype
            device = trans.device
            requires_grad = trans.requires_grad
        else:
            raise ValueError("At least one input argument must be specified")

        if((axis.shape != trans.shape) or
                (axis.shape[:-1]!=angle.shape[:-1]) or (angle.device != trans.device)):
            raise ValueError("Rots and trans incompatible")

        # Force full precision. Happens to the rotations automatically.
        trans = trans.type(torch.float32)

        self._axis = axis.type(torch.float32)
        self._angle = angle.type(torch.float32)
        self._trans = trans

    @staticmethod
    def identity(
        shape: Tuple[int], 
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None, 
        requires_grad: bool = True,
        fmt: str = "quat",
    ):
        """
            Constructs an identity transformation.

            Args:
                shape: 
                    The desired shape
                dtype: 
                    The dtype of both internal tensors
                device: 
                    The device of both internal tensors
                requires_grad: 
                    Whether grad should be enabled for the internal tensors
            Returns:
                The identity transformation
        """
        raise NotImplementedError
        return Rigid(
            Rotation.identity(shape, dtype, device, requires_grad, fmt=fmt),
            identity_trans(shape, dtype, device, requires_grad),
        )

    def __getitem__(self, 
        index: Any,
    ):
        """ 
            Indexes the affine transformation with PyTorch-style indices.
            The index is applied to the shared dimensions of both the rotation
            and the translation.

            E.g.::

                r = Rotation(rot_mats=torch.rand(10, 10, 3, 3), quats=None)
                t = Rigid(r, torch.rand(10, 10, 3))
                indexed = t[3, 4:6]
                assert(indexed.shape == (2,))
                assert(indexed.get_rots().shape == (2,))
                assert(indexed.get_trans().shape == (2, 3))

            Args:
                index: A standard torch tensor index. E.g. 8, (10, None, 3),
                or (3, slice(0, 1, None))
            Returns:
                The indexed tensor 
        """
        # raise NotImplementedError
        if type(index) != tuple:
            index = (index,)
        
        return Rigid_AxisAngle(
            axis=self._axis[index + (slice(None),)],
            angle=self._angle[index + (slice(None),)],
            trans=self._trans[index + (slice(None),)],
        )

    def __mul__(self,
        right: torch.Tensor,
    ):
        """
            Pointwise left multiplication of the transformation with a tensor.
            Can be used to e.g. mask the Rigid.

            Args:
                right:
                    The tensor multiplicand
            Returns:
                The product
        """
        raise NotImplementedError
        if not(isinstance(right, torch.Tensor)):
            raise TypeError("The other multiplicand must be a Tensor")

        new_rots = self._rots * right
        new_trans = self._trans * right[..., None]

        return Rigid(new_rots, new_trans)

    def __rmul__(self,
        left: torch.Tensor,
    ):
        """
            Reverse pointwise multiplication of the transformation with a 
            tensor.

            Args:
                left:
                    The left multiplicand
            Returns:
                The product
        """
        raise NotImplementedError
        return self.__mul__(left)

    @property
    def shape(self) -> torch.Size:
        """
            Returns the shape of the shared dimensions of the rotation and
            the translation.
            
            Returns:
                The shape of the transformation
        """
        s = self._trans.shape[:-1]
        return s

    @property
    def device(self) -> torch.device:
        """
            Returns the device on which the Rigid's tensors are located.

            Returns:
                The device on which the Rigid's tensors are located
        """
        return self._trans.device

    def get_rots(self):
        """
            Getter for the rotation.

            Returns:
                The rotation object
        """
        raise NotImplementedError
        return self._rots

    def get_axis(self):
        """
            Getter for the rotation.

            Returns:
                The rotation object
        """
        return self._axis
    
    def get_angle(self):
        """
            Getter for the rotation.

            Returns:
                The rotation object
        """
        return self._angle % (2 * np.pi)

    def get_trans(self) -> torch.Tensor:
        """
            Getter for the translation.

            Returns:
                The stored translation
        """
        return self._trans

    def get_rot_quats(self):
        """
            Returns the rotation as a quaternion tensor.

            Returns:
                The rotation as a quaternion tensor.
        """
        quat_1 = torch.cos(self._angle / 2) # angle in [0, pi]
        quat_2 = self._axis * torch.sin(self._angle / 2)
        quat = torch.cat([quat_1, quat_2], dim=-1) # https://github.com/scipy/scipy/blob/HEAD/scipy/spatial/transform/_rotation.pyx#L1385-L1402
        # canonical form, w > 0
        flip = (quat[..., :1] < 0).float()
        quat = (-1 * quat) * flip + (1 - flip) * quat    
        return quat

    def compose_q_update_vec(self, 
        q_update_vec: torch.Tensor,
        update_mask: torch.Tensor=None,
        quat4_update=False
    ):
        """
            Composes the transformation with a quaternion update vector of
            shape [*, 6], where the final 6 columns represent the x, y, and
            z values of a quaternion of form (1, x, y, z) followed by a 3D
            translation.

            Args:
                q_vec: The quaternion update vector.
            Returns:
                The composed transformation.
        """
        # quats = self.get_quats()
        # if quat4_update:
        #     # q_update_norm = q_update_vec / q_update_vec.norm(dim=-1,p=2).unsqueeze(-1)
        #     q_update_norm = q_update_vec
        #     quat_update = quat_multiply(quats, q_update_norm)
        # else:
        #     quat_update = quat_multiply_by_vec(quats, q_update_vec)
        # if update_mask is not None:
        #     quat_update = quat_update * update_mask
        # new_quats = quats + quat_update
        # return Rotation(
        #     rot_mats=None, 
        #     quats=new_quats, 
        #     normalize_quats=normalize_quats,
        # )
        q_vec = q_update_vec[..., :4] if quat4_update else q_update_vec[..., :3]
        t_vec = q_update_vec[..., 4:] if quat4_update else q_update_vec[..., 3:]
        # prev_quats = self.get_rot_quats()
        # if quat4_update:
        #     q_update_norm = q_vec / q_vec.norm(dim=-1,p=2).unsqueeze(-1)
        #     quat_update = quat_multiply(prev_quats, q_update_norm)
        # else:
        #     quat_update = quat_multiply_by_vec(prev_quats, q_vec)
        # if update_mask is not None:
        #     quat_update = quat_update * update_mask
        # new_quats = prev_quats + quat_update    
        # new_quats = new_quats / new_quats.norm(dim=-1,p=2).unsqueeze(-1)
        # new_angle = 2 * torch.atan2(
        #     torch.linalg.norm(new_quats[..., 1:], dim=-1),
        #     new_quats[..., 0]
        # ).unsqueeze(-1)
        # new_angle = new_angle % (2 * np.pi)
        # new_axis = new_quats[..., 1:] / torch.linalg.norm(new_quats[..., 1:], dim=-1).unsqueeze(-1)
        quat_update = torch.cat([torch.ones_like(q_vec[..., :1]), q_update_vec[..., :3]], dim=-1)
        quat_update = quat_update / quat_update.norm(dim=-1,p=2).unsqueeze(-1)
        update_rotmat = rotquat_to_rotmat(quat_update)
        new_axis = rot_vec_mul(update_rotmat, self._axis)
        new_angle = self._angle + q_update_vec[..., 3].unsqueeze(-1)
        
        trans_update = t_vec + torch.sin(self._angle) * torch.cross(self._axis, t_vec) + \
            (1-torch.cos(self._angle))*(torch.cross(self._axis, torch.cross(self._axis, t_vec)))
        # trans_update = self._rots.apply(t_vec)
        if update_mask is not None:
            trans_update = trans_update * update_mask
        new_translation = self._trans + trans_update
        return Rigid_AxisAngle(axis=new_axis, angle=new_angle,trans=new_translation)

    def compose_tran_update_vec(self, 
        t_vec: torch.Tensor,
        update_mask: torch.Tensor=None,
    ):
        """
            Composes the transformation with a quaternion update vector of
            shape [*, 3], where columns represent a 3D translation.

            Args:
                q_vec: The quaternion update vector.
            Returns:
                The composed transformation.
        """
        raise NotImplementedError
        trans_update = self._rots.apply(t_vec)
        if update_mask is not None:
            trans_update = trans_update * update_mask
        new_translation = self._trans + trans_update

        return Rigid(self._rots, new_translation)

    def compose(self,
        r,
    ):
        """
            Composes the current rigid object with another.

            Args:
                r:
                    Another Rigid object
            Returns:
                The composition of the two transformations
        """
        new_rot = self._rots.compose_r(r._rots)
        new_trans = self._rots.apply(r._trans) + self._trans
        return Rigid(new_rot, new_trans)

    def compose_r(self,
        rot,
        order='right'
    ):
        """
            Composes the current rigid object with another.

            Args:
                r:
                    Another Rigid object
                order:
                    Order in which to perform rotation multiplication.
            Returns:
                The composition of the two transformations
        """
        raise NotImplementedError
        if order == 'right':
            new_rot = self._rots.compose_r(rot)
        elif order == 'left':
            new_rot = rot.compose_r(self._rots)
        else:
            raise ValueError(f'Unrecognized multiplication order: {order}')
        return Rigid(new_rot, self._trans)

    def apply(self, 
        pts: torch.Tensor,
    ) -> torch.Tensor:
        """
            Applies the transformation to a coordinate tensor.
            with Rodrigues' rotation formula.
            Args:
                pts: A [*, 3] coordinate tensor.
            Returns:
                The transformed points.
        """
        rotated = pts + torch.sin(self._angle) * torch.cross(self._axis, pts,dim=-1) + \
            (1-torch.cos(self._angle))*(torch.cross(self._axis, torch.cross(self._axis, pts)))
        return rotated + self._trans

    def invert_apply(self, 
        pts: torch.Tensor
    ) -> torch.Tensor:
        """
            Applies the inverse of the transformation to a coordinate tensor.

            Args:
                pts: A [*, 3] coordinate tensor
            Returns:
                The transformed points.
        """
        pts = pts - self._trans
        rotated = pts + torch.sin(-self._angle) * torch.cross(self._axis, pts) + \
            (1-torch.cos(-self._angle))*(torch.cross(self._axis, torch.cross(self._axis, pts)))
        return rotated

    def invert(self):
        """
            Inverts the transformation.

            Returns:
                The inverse transformation.
        """
        raise NotImplementedError
        rot_inv = self._rots.invert() 
        trn_inv = rot_inv.apply(self._trans)

        return Rigid(rot_inv, -1 * trn_inv)

    def map_tensor_fn(self, 
        fn
    ):
        """
            Apply a Tensor -> Tensor function to underlying translation and
            rotation tensors, mapping over the translation/rotation dimensions
            respectively.

            Args:
                fn:
                    A Tensor -> Tensor function to be mapped over the Rigid
            Returns:
                The transformed Rigid object
        """     
        raise NotImplementedError
        new_rots = self._rots.map_tensor_fn(fn) 
        new_trans = torch.stack(
            list(map(fn, torch.unbind(self._trans, dim=-1))), 
            dim=-1
        )

        return Rigid(new_rots, new_trans)

    def to_tensor_4x4(self) -> torch.Tensor:
        """
            Converts a transformation to a homogenous transformation tensor.

            Returns:
                A [*, 4, 4] homogenous transformation tensor
        """
        raise NotImplementedError
        tensor = self._trans.new_zeros((*self.shape, 4, 4))
        tensor[..., :3, :3] = self._rots.get_rot_mats()
        tensor[..., :3, 3] = self._trans
        tensor[..., 3, 3] = 1
        return tensor

    @staticmethod
    def from_tensor_4x4(
        t: torch.Tensor
    ):
        """
            Constructs a transformation from a homogenous transformation
            tensor.

            Args:
                t: [*, 4, 4] homogenous transformation tensor
            Returns:
                T object with shape [*]
        """
        raise NotImplementedError
        if(t.shape[-2:] != (4, 4)):
            raise ValueError("Incorrectly shaped input tensor")

        rots = Rotation(rot_mats=t[..., :3, :3], quats=None)
        trans = t[..., :3, 3]
        
        return Rigid(rots, trans)

    def to_tensor_7(self) -> torch.Tensor:
        """
            Converts a transformation to a tensor with 7 final columns, four 
            for the quaternion followed by three for the translation.

            Returns:
                A [*, 7] tensor representation of the transformation
        """
        raise NotImplementedError
        tensor = self._trans.new_zeros((*self.shape, 7))
        tensor[..., :4] = self._rots.get_quats()
        tensor[..., 4:] = self._trans

        return tensor

    @staticmethod
    def from_tensor_7(
        t: torch.Tensor,
        normalize_quats: bool = False,
    ):
        raise NotImplementedError
        if(t.shape[-1] != 7):
            raise ValueError("Incorrectly shaped input tensor")

        quats, trans = t[..., :4], t[..., 4:]

        rots = Rotation(
            rot_mats=None, 
            quats=quats, 
            normalize_quats=normalize_quats
        )

        return Rigid(rots, trans)

    @staticmethod
    def from_3_points(
        p_neg_x_axis: torch.Tensor, 
        origin: torch.Tensor, 
        p_xy_plane: torch.Tensor, 
        eps: float = 1e-8
    ):
        """
            Implements algorithm 21. Constructs transformations from sets of 3 
            points using the Gram-Schmidt algorithm.

            Args:
                p_neg_x_axis: [*, 3] coordinates
                origin: [*, 3] coordinates used as frame origins
                p_xy_plane: [*, 3] coordinates
                eps: Small epsilon value
            Returns:
                A transformation object of shape [*]
        """
        raise NotImplementedError
        p_neg_x_axis = torch.unbind(p_neg_x_axis, dim=-1)
        origin = torch.unbind(origin, dim=-1)
        p_xy_plane = torch.unbind(p_xy_plane, dim=-1)

        e0 = [c1 - c2 for c1, c2 in zip(origin, p_neg_x_axis)]
        e1 = [c1 - c2 for c1, c2 in zip(p_xy_plane, origin)]

        denom = torch.sqrt(sum((c * c for c in e0)) + eps)
        e0 = [c / denom for c in e0]
        dot = sum((c1 * c2 for c1, c2 in zip(e0, e1)))
        e1 = [c2 - c1 * dot for c1, c2 in zip(e0, e1)]
        denom = torch.sqrt(sum((c * c for c in e1)) + eps)
        e1 = [c / denom for c in e1]
        e2 = [
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        ]

        rots = torch.stack([c for tup in zip(e0, e1, e2) for c in tup], dim=-1)
        rots = rots.reshape(rots.shape[:-1] + (3, 3))

        rot_obj = Rotation(rot_mats=rots, quats=None)

        return Rigid(rot_obj, torch.stack(origin, dim=-1))

    def unsqueeze(self, 
        dim: int,
    ):
        """
            Analogous to torch.unsqueeze. The dimension is relative to the
            shared dimensions of the rotation/translation.
            
            Args:
                dim: A positive or negative dimension index.
            Returns:
                The unsqueezed transformation.
        """
        raise NotImplementedError
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")
        rots = self._rots.unsqueeze(dim)
        trans = self._trans.unsqueeze(dim if dim >= 0 else dim - 1)

        return Rigid(rots, trans)

    @staticmethod
    def cat(
        ts,
        dim: int,
    ):
        """
            Concatenates transformations along a new dimension.

            Args:
                ts: 
                    A list of T objects
                dim: 
                    The dimension along which the transformations should be 
                    concatenated
            Returns:
                A concatenated transformation object
        """
        raise NotImplementedError
        rots = Rotation.cat([t._rots for t in ts], dim) 
        trans = torch.cat(
            [t._trans for t in ts], dim=dim if dim >= 0 else dim - 1
        )

        return Rigid(rots, trans)

    def apply_rot_fn(self, fn):
        """
            Applies a Rotation -> Rotation function to the stored rotation
            object.

            Args:
                fn: A function of type Rotation -> Rotation
            Returns:
                A transformation object with a transformed rotation.
        """
        raise NotImplementedError
        return Rigid(fn(self._rots), self._trans)

    def apply_trans_fn(self, fn):
        """
            Applies a Tensor -> Tensor function to the stored translation.

            Args:
                fn: 
                    A function of type Tensor -> Tensor to be applied to the
                    translation
            Returns:
                A transformation object with a transformed translation.
        """
        raise NotImplementedError
        return Rigid(self._rots, fn(self._trans))

    def scale_translation(self, trans_scale_factor: float):
        """
            Scales the translation by a constant factor.

            Args:
                trans_scale_factor:
                    The constant factor
            Returns:
                A transformation object with a scaled translation.
        """
        raise NotImplementedError
        fn = lambda t: t * trans_scale_factor
        return self.apply_trans_fn(fn)

    def stop_rot_gradient(self):
        """
            Detaches the underlying rotation object

            Returns:
                A transformation object with detached rotations
        """
        raise NotImplementedError
        fn = lambda r: r.detach()
        return self.apply_rot_fn(fn)

    @staticmethod
    def make_transform_from_reference(n_xyz, ca_xyz, c_xyz, eps=1e-20):
        """
            Returns a transformation object from reference coordinates.
  
            Note that this method does not take care of symmetries. If you 
            provide the atom positions in the non-standard way, the N atom will 
            end up not at [-0.527250, 1.359329, 0.0] but instead at 
            [-0.527250, -1.359329, 0.0]. You need to take care of such cases in 
            your code.
  
            Args:
                n_xyz: A [*, 3] tensor of nitrogen xyz coordinates.
                ca_xyz: A [*, 3] tensor of carbon alpha xyz coordinates.
                c_xyz: A [*, 3] tensor of carbon xyz coordinates.
            Returns:
                A transformation object. After applying the translation and 
                rotation to the reference backbone, the coordinates will 
                approximately equal to the input coordinates.
        """    
        raise NotImplementedError
        translation = -1 * ca_xyz
        n_xyz = n_xyz + translation
        c_xyz = c_xyz + translation

        c_x, c_y, c_z = [c_xyz[..., i] for i in range(3)]
        norm = torch.sqrt(eps + c_x ** 2 + c_y ** 2)
        sin_c1 = -c_y / norm
        cos_c1 = c_x / norm
        zeros = sin_c1.new_zeros(sin_c1.shape)
        ones = sin_c1.new_ones(sin_c1.shape)

        c1_rots = sin_c1.new_zeros((*sin_c1.shape, 3, 3))
        c1_rots[..., 0, 0] = cos_c1
        c1_rots[..., 0, 1] = -1 * sin_c1
        c1_rots[..., 1, 0] = sin_c1
        c1_rots[..., 1, 1] = cos_c1
        c1_rots[..., 2, 2] = 1

        norm = torch.sqrt(eps + c_x ** 2 + c_y ** 2 + c_z ** 2)
        sin_c2 = c_z / norm
        cos_c2 = torch.sqrt(c_x ** 2 + c_y ** 2) / norm

        c2_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        c2_rots[..., 0, 0] = cos_c2
        c2_rots[..., 0, 2] = sin_c2
        c2_rots[..., 1, 1] = 1
        c1_rots[..., 2, 0] = -1 * sin_c2
        c1_rots[..., 2, 2] = cos_c2

        c_rots = rot_matmul(c2_rots, c1_rots)
        n_xyz = rot_vec_mul(c_rots, n_xyz)

        _, n_y, n_z = [n_xyz[..., i] for i in range(3)]
        norm = torch.sqrt(eps + n_y ** 2 + n_z ** 2)
        sin_n = -n_z / norm
        cos_n = n_y / norm

        n_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        n_rots[..., 0, 0] = 1
        n_rots[..., 1, 1] = cos_n
        n_rots[..., 1, 2] = -1 * sin_n
        n_rots[..., 2, 1] = sin_n
        n_rots[..., 2, 2] = cos_n

        rots = rot_matmul(n_rots, c_rots)

        rots = rots.transpose(-1, -2)
        translation = -1 * translation

        rot_obj = Rotation(rot_mats=rots, quats=None)

        return Rigid(rot_obj, translation)

    def cuda(self):
        """
            Moves the transformation object to GPU memory
            
            Returns:
                A version of the transformation on GPU
        """
        return Rigid_AxisAngle(self._axis.cuda(), self._angle.cuda(), self._trans.cuda())
