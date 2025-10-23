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
from openfold.utils.rigid_utils import quat_multiply, quat_multiply_by_vec
import numpy as np
import torch
import probayes.utils.rotation_conversions as rc



class Rigid_Quat:
    """
        A class representing a rigid transformation. Little more than a wrapper
        around two objects: a Rotation object and a [*, 3] translation
        Designed to behave approximately like a single torch tensor with the 
        shape of the shared batch dimensions of its component parts.
    """
    def __init__(self, 
        quat: torch.Tensor,
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

        if((quat.shape[:-1]!=trans.shape[:-1]) or (quat.device != trans.device)):
            raise ValueError("Rots and trans incompatible")

        # Force full precision. Happens to the rotations automatically.
        trans = trans.type(torch.float32)

        self._quat = quat.type(torch.float32)
        self._trans = trans

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
        
        return Rigid_Quat(
            quat=self._quat[index + (slice(None),)],
            trans=self._trans[index + (slice(None),)],
        )

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

    def get_quat(self):
        """
            Getter for the rotation.

            Returns:
                The rotation object
        """
        return self._quat

    def get_trans(self) -> torch.Tensor:
        """
            Getter for the translation.

            Returns:
                The stored translation
        """
        return self._trans

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
        
        q_vec = q_update_vec[..., :4] if quat4_update else q_update_vec[..., :3]
        # q_vec = torch.randn_like(q_vec)
        t_vec = q_update_vec[..., 4:] if quat4_update else q_update_vec[..., 3:]

        input_quat = self._quat
        if quat4_update: # TODO: do we need 4 variables?
            quat_update = quat_multiply(input_quat, q_vec)
        else:
            quat_update = quat_multiply_by_vec(input_quat, q_vec)
        # quat_update is antipodal equivariant
        if update_mask is not None:
            quat_update = quat_update * update_mask
        new_quats = input_quat + quat_update 
        new_quats = new_quats / new_quats.norm(dim=-1,keepdim=True) 
        # new_quats = input_quat + input_quat * quat_update
        
        # update translation
        trans_update = rc.quaternion_apply(self._quat, t_vec)
        if update_mask is not None:
            trans_update = trans_update * update_mask
        new_translation = self._trans + trans_update

        return Rigid_Quat(new_quats, new_translation)

    def apply(self, 
        pts: torch.Tensor,
    ) -> torch.Tensor:
        """
            Applies the quaternion based transformation to a coordinate tensor.
            with Rodrigues' rotation formula.
            Args:
                pts: A [*, 3] coordinate tensor.
            Returns:
                The transformed points.
        """
        # rotated = pts + torch.sin(self._angle) * torch.cross(self._axis, pts,dim=-1) + \
        #     (1-torch.cos(self._angle))*(torch.cross(self._axis, torch.cross(self._axis, pts)))
        rotated = rc.quaternion_apply(self._quat, pts)
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
        inverse_quat = rc.quaternion_invert(self._quat)
        rotated = rc.quaternion_apply(inverse_quat, pts)        
        # rotated = pts + torch.sin(-self._angle) * torch.cross(self._axis, pts) + \
        #     (1-torch.cos(-self._angle))*(torch.cross(self._axis, torch.cross(self._axis, pts)))
        return rotated

    def cuda(self):
        """
            Moves the transformation object to GPU memory
            
            Returns:
                A version of the transformation on GPU
        """
        # return Rigid_Quat(self._axis.cuda(), self._angle.cuda(), self._trans.cuda())
        return Rigid_Quat(self._quat.cuda(), self._trans.cuda())
    
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
        return Rigid_Quat(self._quat, fn(self._trans))
