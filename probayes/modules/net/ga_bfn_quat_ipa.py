import torch
from torch import nn
from probayes.core import ipa_pytorch as ipa_pytorch
from probayes.data import utils as du
from probayes.core.utils import get_index_embedding, get_time_embedding
from probayes.modules.protein.constants import ANG_TO_NM_SCALE, NM_TO_ANG_SCALE
from probayes.modules.common.layers import AngularEncoding
from probayes.utils.quat_rigid_utils import Rigid_Quat
import probayes.utils.rotation_conversions as rc
import math


class GAEncoder_BFN(nn.Module):
    def __init__(self, ipa_conf, bfn_conf):
        super().__init__()
        self._ipa_conf = ipa_conf 
        self._bfn_conf = bfn_conf
        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12) # 25*5=120, for competitive embedding size
        
        self.angle_fourer_feat = 'angle_fourier_feat' in self._ipa_conf.keys() and self._ipa_conf.angle_fourier_feat
        self.only_psi = 'only_psi' in self._bfn_conf.keys() and self._bfn_conf.only_psi
        self.ipa_with_sc = True if 'ipa_with_sc' in self._ipa_conf.keys() and self._ipa_conf.ipa_with_sc else False
        
        if self.angle_fourer_feat:
            self.angle_net = nn.Sequential(
                nn.Linear(self._ipa_conf.c_s+self.angles_embedder.get_out_dim(in_dim=5)+5, self._ipa_conf.c_s),nn.ReLU(),
                nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
                nn.Linear(self._ipa_conf.c_s, 5)
            )
        else:
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
        )
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # mixer
        n_acc_dim = 1 if self._bfn_conf.cond_acc else 0 # quaternion accuracy
        if self._bfn_conf.remov_ang_in:
            n_angle_dim = 0
        else:
            n_angle_dim = self.angles_embedder.get_out_dim(in_dim=1)+1
            if 'ablation_shortcut' in self._bfn_conf.keys() and self._bfn_conf.ablation_shortcut:
                n_angle_dim = n_angle_dim * 5
                
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + n_acc_dim + n_angle_dim, self._ipa_conf.c_s), 
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # incoporate sc angle features
        if self.ipa_with_sc: 
            self.bb_update_sc_mixer = nn.Sequential(
                nn.Linear(self.angles_embedder.get_out_dim(in_dim=5)+5, self._ipa_conf.c_s), 
                nn.ReLU(),
                nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
                nn.ReLU(),
                nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
                # nn.LayerNorm(self._ipa_conf.c_s),                
            )
            self.bb_update_sc = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)
            self.bb_sc_ln = nn.LayerNorm(self._ipa_conf.c_s)
            
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

    def forward(self, t, quat_t, trans_t, angles_t, seqs_t, node_embed, edge_embed, generate_mask, res_mask, acc_ang_t, acc_quat_t):
        num_batch, num_res, num_classes = seqs_t.shape
        seqs_t = 2 * seqs_t - 1  # Convert to -1, 1
        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        
        if self._bfn_conf.remov_ang_in:
            node_embed = self.res_feat_mixer(torch.cat([node_embed, 
                                                    self.current_seq_embedder_smooth(seqs_t), 
                                                    self.embed_t(t,node_mask),
                                                    2*acc_quat_t-1
                                                    ],
                                                dim=-1))
        else:
            if 'ablation_shortcut' in self._bfn_conf.keys() and self._bfn_conf.ablation_shortcut:
                angle_feat = self.angles_embedder(angles_t)
                acc_angle = acc_ang_t
            else:
                angle_feat = self.angles_embedder(angles_t[..., :1])
                acc_angle = acc_ang_t[..., :1]

            angle_feat_cat = torch.cat([angle_feat,acc_angle],dim=-1)
            node_embed = self.res_feat_mixer(torch.cat([
                                                    node_embed, 
                                                    self.current_seq_embedder_smooth(seqs_t), 
                                                    self.embed_t(t,node_mask),
                                                    2*acc_quat_t-1,
                                                    angle_feat_cat
                                                    ],dim=-1))

        node_embed = node_embed * node_mask[..., None]
        curr_rigids = Rigid_Quat(quat=quat_t, trans=trans_t)
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
        if self.ipa_with_sc:
            angle_embed = torch.cat([self.angles_embedder(angles_t), 2*acc_ang_t-1], dim=-1)
            bb_sc_node_embed = self.bb_update_sc_mixer(angle_embed) + node_embed
            bb_sc_node_embed = self.bb_sc_ln(bb_sc_node_embed)
            bb_update = self.bb_update_sc(bb_sc_node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(bb_update, node_mask[..., None])
        
        pred_trans1 = curr_rigids.get_trans()
        pred_seqs1_prob = self.seq_net(node_embed)
        
        if self.angle_fourer_feat:
            angles_embed = self.angles_embedder(angles_t)
            concat_angle_feature = torch.cat([angles_embed,node_embed,acc_ang_t],dim=2)
            pred_angles1 = self.angle_net(concat_angle_feature) + angles_t
            pred_angles1 = pred_angles1 % (2*math.pi) # between (0,2pi)
        else:
            pred_angles1 = self.angle_net(node_embed) + angles_t
            pred_angles1 = pred_angles1 % (2*math.pi) # between (0,2pi)

        pred_quat = curr_rigids.get_quat()
        return pred_quat.float(), pred_trans1, pred_angles1, pred_seqs1_prob.softmax(dim=-1)