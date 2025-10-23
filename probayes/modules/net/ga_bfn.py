import torch
from torch import nn

from probayes.core import ipa_pytorch as ipa_pytorch
from probayes.data import utils as du

from probayes.core.utils import get_index_embedding, get_time_embedding

from probayes.modules.protein.constants import ANG_TO_NM_SCALE, NM_TO_ANG_SCALE
from probayes.modules.common.layers import AngularEncoding
from ema_pytorch import EMA
import math


class GAEncoder_BFN(nn.Module):
    def __init__(self, ipa_conf, bfn_conf):
        super().__init__()
        self._ipa_conf = ipa_conf 
        self._bfn_conf = bfn_conf
        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12) # 25*5=120, for competitive embedding size
        # self.angle_net = nn.Sequential(
        #     nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
        #     nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
        #     nn.Linear(self._ipa_conf.c_s, 5)
        #     # nn.Linear(self._ipa_conf.c_s, 22)
        # )
        angle_input_dim = self.angles_embedder.get_out_dim(in_dim=5)+5 if self._bfn_conf.cond_acc else self.angles_embedder.get_out_dim(in_dim=5)
        self.angle_concat_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s+angle_input_dim, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 5)
            # nn.Lilf._ipa_conf.c_s, 22)
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

        # mixer
        n_added_dim = self.angles_embedder.get_out_dim(in_dim=5) if not self._bfn_conf.remov_ang_in else 0
        n_added_dim += 2 if self._bfn_conf.use_quat else 0 # rotation acc
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + n_added_dim, self._ipa_conf.c_s), 
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
                self._ipa_conf.c_s, use_rot_updates=True, quat4_update=self._ipa_conf.quat4_update)

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

    def forward(self, t, rotmats_t, trans_t, angles_t, seqs_t, node_embed, edge_embed, generate_mask, res_mask, log_acc_t, quat_acc=None):
        num_batch, num_res, num_classes = seqs_t.shape
        seqs_t = 2 * seqs_t - 1  # Convert to -1, 1
        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        
        mixed_feature = torch.cat([node_embed, 
                                        self.current_seq_embedder_smooth(seqs_t), 
                                        self.embed_t(t,node_mask)],
                                        dim=-1)
        if not self._bfn_conf.remov_ang_in:
            mixed_feature = torch.cat([mixed_feature,self.angles_embedder(angles_t).reshape(num_batch,num_res,-1)],dim=-1)
        if (not self._bfn_conf.remov_ang_in) and self._bfn_conf.cond_acc:
            mixed_feature = torch.cat([mixed_feature,log_acc_t.float()],dim=-1)
        if self._bfn_conf.use_quat:
            mixed_feature = torch.cat([mixed_feature,quat_acc.float()],dim=-1)
        
        node_embed = self.res_feat_mixer(mixed_feature)
        node_embed = node_embed * node_mask[..., None]
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
            node_embed = self.trunk[f'node_transition_{b}'](node_embed) # Res Connected MLP
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None],quat4_update=self._ipa_conf.quat4_update)

            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]
        
        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        # choice 1: no residual connection
        # pred_angles1 = (self.angle_net(node_embed) + angles_t / torch.pi) % 2   
        # pred_angles1 = pred_angles1 * torch.pi
        # choice 2: simple out, no featurized
        # pred_angles1 = self.angle_net(node_embed) + angles_t
        # choice 3: use angles embedding
        angles_embed = self.angles_embedder(angles_t).reshape(num_batch,num_res,-1)
        concat_angle_feature = torch.cat([angles_embed,node_embed],dim=2)
        if self._bfn_conf.cond_acc:
            concat_angle_feature = torch.cat([angles_embed,node_embed,log_acc_t],dim=2)
        else:
            concat_angle_feature = torch.cat([angles_embed,node_embed],dim=2)
        pred_angles1 = self.angle_concat_net(concat_angle_feature) + angles_t
        pred_angles1 = pred_angles1 % (2*math.pi) # inductive bias to bound between (0,2pi)
        
        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob.softmax(dim=-1)