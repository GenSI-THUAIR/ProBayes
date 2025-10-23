import torch
from torch import nn

from probayes.core import ipa_pytorch as ipa_pytorch
from probayes.data import utils as du

from probayes.core.utils import get_index_embedding, get_time_embedding

from probayes.modules.protein.constants import ANG_TO_NM_SCALE, NM_TO_ANG_SCALE
from probayes.modules.common.layers import AngularEncoding
from probayes.modules.protein.constants import sc_rg_vocab

import math

def build_mlp(in_dim,out_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),nn.ReLU(),
        nn.Linear(hidden_dim, out_dim)
    )

class SinusoidalPositionEmbedding(nn.Module):
    """
    Sin-Cos Positional Embedding
    """
    def __init__(self, output_dim):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim

    def forward(self, position_ids):
        device = position_ids.device
        position_ids = position_ids[None] # [1, N]
        indices = torch.arange(self.output_dim // 2, device=device, dtype=torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.reshape(-1, self.output_dim)
        return embeddings

class GAEncoder(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf 
        
        self.Ks = {k:len(v) for k,v in sc_rg_vocab.items()}
        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12) # 25*5=120, for competitive embedding size
        # self.angle_net = nn.Sequential(
        #     nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
        #     nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
        #     nn.Linear(self._ipa_conf.c_s, 5)
        #     # nn.Linear(self._ipa_conf.c_s, 22)
        # )

        # for condition on current seq
        # self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)
        self.ar_time_embedder = SinusoidalPositionEmbedding(output_dim=self._ipa_conf.c_s)
        self.sg_gen_embedder = nn.ModuleDict({
            '0': nn.Embedding(22, self._ipa_conf.c_s),
            '1': nn.Embedding(22, self._ipa_conf.c_s),
            '2': nn.Embedding(22, self._ipa_conf.c_s),
            '3': nn.Embedding(22, self._ipa_conf.c_s),
        })
        self.sg_context_embedder = nn.ModuleDict({
            '0': nn.Embedding(22, self._ipa_conf.c_s),
            '1': nn.Embedding(22, self._ipa_conf.c_s),
            '2': nn.Embedding(22, self._ipa_conf.c_s),
            '3': nn.Embedding(22, self._ipa_conf.c_s),
        })
        self.sg_context_feat = nn.Sequential(# 暂时去掉angular embedding
            nn.Linear(4 * self._ipa_conf.c_s, self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )
        self.rg_nets = nn.ModuleDict({
            str(ar_time): build_mlp(self._ipa_conf.c_s, self.Ks[ar_time+1], self._ipa_conf.c_s) for ar_time in range(4)
        })
        # self.seq_net = nn.Sequential(
        #     nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
        #     nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
        #     nn.Linear(self._ipa_conf.c_s, 20)
        #     # nn.Linear(self._ipa_conf.c_s, 22)
        # )
        # self.angle_concat_net = nn.Sequential(
        #     nn.Linear(self._ipa_conf.c_s+self.angles_embedder.get_out_dim(in_dim=5), self._ipa_conf.c_s),nn.ReLU(),
        #     nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
        #     nn.Linear(self._ipa_conf.c_s, 5)
        #     # nn.Linear(self._ipa_conf.c_s, 22)
        # )
        self.angle_out_net = build_mlp(self._ipa_conf.c_s, 2, self._ipa_conf.c_s)
        # mixer
        self.res_feat_mixer = nn.Sequential(# 暂时去掉angular embedding
            nn.Linear(5 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=6), self._ipa_conf.c_s),
            # nn.Linear(3 * self._ipa_conf.c_s, self._ipa_conf.c_s),
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

    def forward(self, t, rotmats_t, trans_t, 
                bb_angles_t, sc_angles_gen_t, sc_angles_context, 
                rg_gen_t, rg_context, 
                node_embed, edge_embed, generate_mask, res_mask, ar_time):
        num_batch, num_res = rg_gen_t.shape

        # incorperate timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        
        # incorporate noised seq info
        rg_gen_noised_embed = self.sg_gen_embedder[str(ar_time)](rg_gen_t)
        rg_context_embed = self.sg_context_embedder[str(ar_time)](rg_context).reshape(num_batch,num_res,-1)
        rg_context_feat = self.sg_context_feat(rg_context_embed)
        
        bb_angle_feat = self.angles_embedder(bb_angles_t).reshape(num_batch,num_res,-1)
        sc_angle_context_feat = self.angles_embedder(sc_angles_context).reshape(num_batch,num_res,-1)
        sc_angle_gen_feat = self.angles_embedder(sc_angles_gen_t).reshape(num_batch,num_res,-1)
        
        ar_time_input = torch.tensor([ar_time]*num_batch).to(rg_gen_t.device)
        ar_time_input = self.ar_time_embedder(ar_time_input).unsqueeze(1).repeat(1,num_res,1)
        node_embed = self.res_feat_mixer(torch.cat([node_embed, 
                                                    rg_gen_noised_embed, 
                                                    rg_context_feat,
                                                    bb_angle_feat,
                                                    sc_angle_context_feat,
                                                    sc_angle_gen_feat,
                                                    self.embed_t(t,node_mask),
                                                    ar_time_input
                                                    ],dim=-1))
        # node_embed = self.res_feat_mixer(torch.cat([node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t,node_mask), self.angles_embedder(angles_t).reshape(num_batch,num_res,-1)],dim=-1))
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
        pred_trans_1 = curr_rigids.get_trans()
        pred_rotmats_1 = curr_rigids.get_rots().get_rot_mats()
        pred_rg_prob = self.rg_nets[str(ar_time)](node_embed)
        # pred_angles1 = self.angle_net(node_embed)
        # angles_embed = self.angles_embedder(angles_t).reshape(num_batch,num_res,-1)
        # concat_angle_feature = torch.cat([angles_embed,node_embed],dim=2)
        pred_angles1 = self.angle_out_net(node_embed)
        pred_bb_angle, pred_sc_angle = pred_angles1.unbind(dim=-1)
        pred_bb_angle = (pred_bb_angle + bb_angles_t) % (2*math.pi)
        pred_sc_angle = (pred_sc_angle + sc_angles_gen_t) % (2*math.pi)       
        return pred_rotmats_1, pred_trans_1, pred_rg_prob, pred_bb_angle, pred_sc_angle