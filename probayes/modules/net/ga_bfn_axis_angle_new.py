import torch
from torch import nn

from probayes.core import ipa_pytorch as ipa_pytorch
from probayes.data import utils as du

from probayes.core.utils import get_index_embedding, get_time_embedding

from probayes.modules.protein.constants import ANG_TO_NM_SCALE, NM_TO_ANG_SCALE
from probayes.modules.common.layers import AngularEncoding
from probayes.utils import rotation_conversions as rc

from ema_pytorch import EMA
import math


class GAEncoder_BFN(nn.Module):
    def __init__(self, ipa_conf, bfn_conf):
        super().__init__()
        self._ipa_conf = ipa_conf 
        self._bfn_conf = bfn_conf
        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12) # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 5)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for condition on current seq
        # self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)
        # self.current_seq_embedder_smooth = nn.Linear(20, self._ipa_conf.c_s) # bfn previous
        self.current_seq_embedder_smooth = nn.Sequential(
            nn.Linear(20, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )
        
        self.pred_rotmat = self._bfn_conf.pred_rotmat
        if not self.pred_rotmat:
            # axis and angle out
            self.axis_angle_net = nn.Sequential(
                nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
                nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
                nn.Linear(self._ipa_conf.c_s, 3 + 1)
            )

        # mixer
        n_acc_dim = 5  if self._bfn_conf.cond_acc else 0
        n_acc_dim = - self.angles_embedder.get_out_dim(in_dim=5) if self._bfn_conf.remov_ang_in else 0
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=5) + n_acc_dim + 2, self._ipa_conf.c_s), # 5 is the dimension of log_acc_t
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks-1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )
    
    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotaxis_t, acc_rot_axis, rotang_t, acc_rot_ang, trans_t, angles_t, seqs_t, node_embed, edge_embed, generate_mask, res_mask, log_acc_t):
        num_batch, num_res, num_classes = seqs_t.shape
        seqs_t = 2 * seqs_t - 1  # Convert to -1, 1
        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        
        if self._bfn_conf.remov_ang_in:
            node_embed = self.res_feat_mixer(torch.cat([node_embed, 
                                                    self.current_seq_embedder_smooth(seqs_t), 
                                                    acc_rot_axis,
                                                    acc_rot_ang,
                                                    self.embed_t(t,node_mask)],
                                                dim=-1))
        # else:
        #     if self._bfn_conf.cond_acc:
        #         node_embed = self.res_feat_mixer(torch.cat([node_embed, 
        #                                                     self.current_seq_embedder_smooth(seqs_t), 
        #                                                     self.embed_t(t,node_mask), 
        #                                                     log_acc_t.float(),
        #                                                     self.angles_embedder(angles_t).reshape(num_batch,num_res,-1)],
        #                                                 dim=-1))
        #     else:
        #         node_embed = self.res_feat_mixer(torch.cat([node_embed, 
        #                                                     self.current_seq_embedder_smooth(seqs_t), 
        #                                                     self.embed_t(t,node_mask), 
        #                                                     self.angles_embedder(angles_t).reshape(num_batch,num_res,-1)],
        #                                                 dim=-1))
        node_embed = node_embed * node_mask[..., None]
        rotmats_t = rc.axis_angle_to_matrix(rotaxis_t*rotang_t)
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]
        
        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        
        if self.pred_rotmat:
            pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        else:
            # pred axis angle   
            pred_axis_angle = self.axis_angle_net(node_embed)
            pred_rot_axis1 = pred_axis_angle[..., :3] + rotaxis_t.float()
            pred_rot_axis1 = pred_rot_axis1 / torch.norm(pred_rot_axis1, dim=-1, keepdim=True)
            pred_rot_angle1 = pred_axis_angle[..., 3:] + rotang_t
            pred_rot_angle1 = (pred_rot_angle1 % (2*math.pi))    
        
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles1 = self.angle_net(node_embed) + angles_t
        pred_angles1 = pred_angles1 % (2*math.pi) # inductive bias to bound between (0,2pi)
        
        if self.pred_rotmat:
            return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob.softmax(dim=-1)
        else:
            return pred_rot_axis1, pred_rot_angle1, pred_trans1, pred_angles1, pred_seqs1_prob.softmax(dim=-1)