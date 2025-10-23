import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import copy
import math
from tqdm.auto import tqdm
import functools
import os
import argparse
import pandas as pd
from copy import deepcopy

from probayes.utils.train import recursive_to

from probayes.modules.common.geometry import reconstruct_backbone, reconstruct_backbone_partially, align, batch_align
from probayes.modules.protein.writers import save_pdb

from probayes.utils.data import PaddingCollate

from probayes.core.utils import process_dic

from probayes.core.flow_model import FlowModel
from probayes.core.bfn_model_quat import BFNModel_quat
from probayes.core.torsion import full_atom_reconstruction, get_heavyatom_mask
from probayes.eval.geometry import get_side_chain_metrics
from probayes.core.torsion import get_torsion_angle, torsions_mask
# from probayes.modules.protein.parsers import parse_pdb
# from remote.RDE_PPI.rde.utils.protein.parsers import parse_pdb
from probayes.data.parsers import parse_pdb

from Bio.PDB import PDBExceptions
from probayes.data import all_atom
from remote.ppflow.tools.relax.rosetta_packing import side_chain_packing
from remote.PepGLAD.data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from probayes.modules.protein.constants import AA,resindex_to_ressymb
from probayes.data import residue_constants
from probayes.data import utils as du

ATOM_MASK = torch.tensor(residue_constants.restype_atom14_mask)
collate_fn = PaddingCollate(eight=False)
from probayes.modules.protein.constants import (
    BBHeavyAtom, 
)
import argparse


def item_to_batch(item, nums=32):
    data_list = [deepcopy(item) for i in range(nums)]
    return collate_fn(data_list)


def _mask_select(v, mask):
    if isinstance(v, str):
        return ''.join([s for i, s in enumerate(v) if mask[i]])
    elif isinstance(v, list):
        return [s for i, s in enumerate(v) if mask[i]]
    elif isinstance(v, torch.Tensor):
        return v[mask]
    else:
        return v

def sample_from_data(data, model:FlowModel, device, num_steps=200, num_samples=1, sample_bb=True,sample_ang=True,sample_seq=True):
    batch = recursive_to(item_to_batch(data, nums=num_samples),device=device)
    traj = model.sample(batch, num_steps=num_steps,sample_bb=sample_bb,sample_ang=sample_ang,sample_seq=sample_seq)
    final = recursive_to(traj[-1], device=device)
    pos_bb = reconstruct_backbone(R=final['rotmats'],t=final['trans'],aa=final['seqs'],chain_nb=batch['chain_nb'],res_nb=batch['res_nb'],mask=batch['res_mask']) # (32,L,4,3)
    
    pos_ha = F.pad(pos_bb, pad=(0,0,0,15-4), value=0.) # (32,L,A,3) pos14 A=14
    pos_new = torch.where(batch['generate_mask'][:,:,None,None],pos_ha,batch['pos_heavyatom'])
    mask_bb_atoms = torch.zeros_like(batch['mask_heavyatom'])
    mask_bb_atoms[:,:,:4] = True
    mask_new = torch.where(batch['generate_mask'][:,:,None],mask_bb_atoms,batch['mask_heavyatom'])
    aa_new = final['seqs']
    
    side_angles_metrics = get_side_chain_metrics(final['angles_1'], final['angles'], data['generate_mask'],data['torsion_angle_mask'])
    
    chain_nb = torch.LongTensor([0 if gen_mask else 1 for gen_mask in data['generate_mask']])
    chain_id = ['B' if gen_mask else 'A' for gen_mask in data['generate_mask']]
    icode = [' ' for _ in range(len(data['icode']))]
    # ref_saved = {
    #                 'chain_nb':chain_nb,'chain_id':chain_id,'resseq':data['resseq'],'icode':data['icode'],
    #                 'aa':data['aa'], 'mask_heavyatom':data['mask_heavyatom'], 'pos_heavyatom':data['pos_heavyatom'],
    #             }
    # mask select the reference peptide
    peptide_mask = data['generate_mask']
    ref_resseq = _mask_select(data['resseq'],peptide_mask)
    ref_icode = _mask_select(data['icode'],peptide_mask)
    ref_chain_id = _mask_select(data['chain_id'],peptide_mask)
    ref_chain_nb = _mask_select(data['chain_nb'],peptide_mask)
    ref_aa = _mask_select(data['aa'],peptide_mask)
    ref_mask = _mask_select(data['mask_heavyatom'],peptide_mask)
    ref_pos = _mask_select(data['pos_heavyatom'],peptide_mask)
    ref_saved = {
                    'chain_nb':ref_chain_nb,'chain_id':ref_chain_id,
                    'resseq':ref_resseq,'icode':ref_icode,
                    'aa':ref_aa, 'mask_heavyatom':ref_mask, 'pos_heavyatom':ref_pos,
                }    
    # get the peptide chain_id and chain_nb
    pep_chain_nb = ref_chain_nb.unique()
    pep_chain_id = set(ref_chain_id)
    assert len(pep_chain_nb) == 1 and len(pep_chain_id) == 1, 'Only support single peptide chain'
    ref_merge = save_pdb(data,path=None)
    sample_merges = []
    ref_pep_structure = save_pdb(ref_saved,path= None)   
    sample_pep_structures = []
    for i in range(num_samples):
        # ref_bb_pos = data['pos_heavyatom'][i][:,:4].cpu()
        # pred_bb_pos = pos_new[i][:,:4].cpu()
        sample_pep_aa = _mask_select(aa_new[i],peptide_mask)
        sample_pep_mask = _mask_select(mask_new[i],peptide_mask)
        sample_pep_pos = _mask_select(pos_new[i],peptide_mask)
        pep_saved = {
                      'chain_nb':ref_chain_nb,'chain_id':ref_chain_id,
                      'resseq':ref_resseq,'icode':ref_icode,
                      'aa':sample_pep_aa.cpu(), 'mask_heavyatom':sample_pep_mask.cpu(),
                      'pos_heavyatom':sample_pep_pos.cpu(),
                    }
        sample_pep_structures.append(save_pdb(pep_saved,path = None))
        merge_saved = {
            'chain_nb':data['chain_nb'],'chain_id':data['chain_id'],'resseq':data['resseq'],'icode':data['icode'],
            'aa':aa_new[i].cpu(), 'mask_heavyatom':mask_new[i].cpu(), 'pos_heavyatom':pos_new[i].cpu(),
            }
        sample_merges.append(save_pdb(merge_saved,path=None))
    return sample_pep_structures, ref_pep_structure, sample_merges, ref_merge, side_angles_metrics

import pickle as pkl

def generate(ckpt_path, data, model, device, 
             num_steps, num_samples, 
             sample_bb,sample_ang,sample_seq,
             dataset_pdb_dir, sample_mode,
             sc_pack='rosetta'):
    print('sample_bb:', sample_bb,'sample_ang:', sample_ang, 'sample_seq:', sample_seq)
    
    if (not sample_bb) and (not sample_seq) and (sample_ang):
        sc_pack = 'generated' # sidechain packing mode

    if (sample_bb) and (not sample_seq) and (sample_ang):
        sc_pack = 'generated' # folding/binding comformation generation mode
        
    # sc_pack = 'generated'
    print('sidechain packing mode', sc_pack)
    
    # sample the trajectory
    batch = recursive_to(item_to_batch(data, nums=num_samples),device=device)
    
    # newest dataset pipeline
    assert 'chi' in batch.keys()
    
    traj = model.sample(batch, num_steps=num_steps,sample_bb=sample_bb,sample_ang=sample_ang,sample_seq=sample_seq,mode=sample_mode)
    final_state = recursive_to(traj[-1], device=device)
    
    if sc_pack == 'ground_truth':
        # reconstruct the backbone structure
        pos_bb = reconstruct_backbone(
                R=final_state['rotmats'],
                t=final_state['trans'],
                aa=final_state['seqs'],
                chain_nb=batch['chain_nb'],
                res_nb=batch['res_nb'],mask=batch['res_mask']) # (num_samples,seq_len,num_bb_atoms=4,n_dim=3)
        # pad the sidechain atoms coordinates as 0.
        pos_ha = F.pad(pos_bb, pad=(0,0,0,15-4), value=0.) # (32,L,A,3) pos14 A=14
        # only substitute backbone atoms, sidechain atoms use the ground truth coordinates
        pos_new = torch.where(batch['generate_mask'][:,:,None,None],pos_ha,batch['pos_heavyatom']) 
        mask_bb_atoms = torch.zeros_like(batch['mask_heavyatom'])
        mask_bb_atoms[:,:,:4] = True
        mask_new = torch.where(batch['generate_mask'][:,:,None],mask_bb_atoms,batch['mask_heavyatom'])
    elif sc_pack == 'rosetta':
        bb_atoms = all_atom.to_atom37(trans=final_state['trans'], rots=final_state['rotmats'])[:, :, :3] 
        pos_ha = F.pad(bb_atoms, pad=(0,0,0,15-3), value=0.) # (32,L,A,3) pos14 A=14
        pos_new = torch.where(batch['generate_mask'][:,:,None,None],pos_ha,batch['pos_heavyatom']) 
        mask_bb_atoms = torch.zeros_like(batch['mask_heavyatom'])
        mask_bb_atoms[:,:,:3] = True
        mask_new = torch.where(batch['generate_mask'][:,:,None],mask_bb_atoms,batch['mask_heavyatom'])   
    elif sc_pack == 'generated':
        pos_ha = all_atom.compute_allatom(
                        bb_rigids=du.create_rigid(rots=final_state['rotmats'],trans=final_state['trans']),
                        angles=final_state['angles'],aatype=final_state['seqs'],)

        pos_new = F.pad(pos_ha, pad=(0,0,0,15-14), value=0.)
        mask_bb_atoms = torch.zeros_like(batch['mask_heavyatom'])
        mask_bb_atoms[:,:,:14] = True
        mask_new = torch.where(batch['generate_mask'][:,:,None],mask_bb_atoms,batch['mask_heavyatom'])

    aa_gen = final_state['seqs']

    
    pdb_id = data['id']
    
    # mask select the reference peptide
    peptide_mask = data['generate_mask']
    ref_resseq = _mask_select(data['resseq'],peptide_mask)
    ref_icode = _mask_select(data['icode'],peptide_mask)
    ref_chain_id = _mask_select(data['chain_id'],peptide_mask)
    ref_chain_nb = _mask_select(data['chain_nb'],peptide_mask)
    ref_aa = _mask_select(data['aa'],peptide_mask)
    ref_mask = _mask_select(data['mask_heavyatom'],peptide_mask)
    ref_pos = _mask_select(data['pos_heavyatom'],peptide_mask)
    ref_saved = {
                    'chain_nb':ref_chain_nb,'chain_id':ref_chain_id,
                    'resseq':ref_resseq,'icode':ref_icode,
                    'aa':ref_aa, 'mask_heavyatom':ref_mask, 'pos_heavyatom':ref_pos,
                }    
    
    # get the peptide chain_id and chain_nb
    pep_chain_nb = ref_chain_nb.unique()
    pep_chain_id = set(ref_chain_id)
    assert len(pep_chain_nb) == 1 and len(pep_chain_id) == 1, 'Only support single peptide chain'
    pep_chain_id = pep_chain_id.pop()
    receptor_chain_id = set(_mask_select(data['chain_id'],~peptide_mask))
    
    if len(receptor_chain_id) > 1:
        print(f'Warning: Multiple receptor chains detected')
        raise PDBExceptions.PDBConstructionException('Multiple receptor chains detected')
    receptor_chain_id = receptor_chain_id.pop()
    
    # create related directories
    save_path = os.path.join(ckpt_path,'results')
    # os.mkdir(save_path) if not os.path.exists(save_path) else None
    os.mkdir(os.path.join(save_path,'references')) if not os.path.exists(os.path.join(save_path,'references')) else None
    os.mkdir(os.path.join(save_path,'candidates')) if not os.path.exists(os.path.join(save_path,'candidates')) else None
    os.mkdir(os.path.join(save_path,'candidates',f'{pdb_id}')) if not os.path.exists(os.path.join(save_path,'candidates',f'{pdb_id}')) else None
    
    ref_merge = save_pdb(data,path=os.path.join(save_path,'references',f'{pdb_id}_ref.pdb'))
    ref_lig_path = os.path.join(save_path,'references',f'{pdb_id}_ref_lig.pdb')
    # ref_pep_structure = save_pdb(ref_saved,path=None)  
    save_pdb(ref_saved,path=ref_lig_path)   
    
    # sample_pep_structures = []
    x_pkl_files = []
    gen_pep_seqs = []
    gen_lig_paths = []
    
    for i in range(num_samples):
        sample_pep_aa = _mask_select(aa_gen[i],peptide_mask)
        sample_pep_mask = _mask_select(mask_new[i],peptide_mask)
        sample_pep_pos = _mask_select(pos_new[i],peptide_mask)

        sample_pep_seq = ''.join([resindex_to_ressymb[v] for v in sample_pep_aa.cpu().tolist()])
        gen_pep_seqs.append(sample_pep_seq)
        x_pkl_file = os.path.join(save_path, pdb_id + f'_gen_{i}_X.pkl')
        
        # gen_lig_paths = []
        
        if sc_pack == 'rosetta':
            # peptide backbone to be packed 
            pep_backbone_path = os.path.join(save_path,'candidates',f'{pdb_id}',f'{pdb_id}_pep_bb3_{i}.pdb')
            pep_backbone_saved = {
                        'chain_nb':ref_chain_nb,'chain_id':ref_chain_id,
                        'resseq':ref_resseq,'icode':ref_icode,
                        'aa':sample_pep_aa.cpu(), 'mask_heavyatom':sample_pep_mask.cpu(),
                        'pos_heavyatom':sample_pep_pos[:,:3,:].cpu(),
                        } 
            save_pdb(pep_backbone_saved,path=pep_backbone_path)
            packed_out_file = side_chain_packing(pdb_file=pep_backbone_path, 
                                                output_file=os.path.join(save_path,'candidates',f'{pdb_id}',f'{pdb_id}_pep_packed_{i}.pdb'))
            gen_lig_paths.append(packed_out_file)
            packed_pep = parse_pdb(path=packed_out_file)[0]
            # save coordinates
            pkl.dump(packed_pep['pos_heavyatom'].cpu().tolist(), open(x_pkl_file, 'wb')) # check here
        elif sc_pack == 'generated':
            pep_path = os.path.join(save_path,'candidates',f'{pdb_id}',f'{pdb_id}_pep_atom14_{i}.pdb')
            pep_saved = {
                        'chain_nb':ref_chain_nb,'chain_id':ref_chain_id,
                        'resseq':ref_resseq,'icode':ref_icode,
                        'aa':sample_pep_aa.cpu(), 'mask_heavyatom':sample_pep_mask.cpu(),
                        'pos_heavyatom':sample_pep_pos[:,:14,:].cpu(),
                        } 
            save_pdb(pep_saved,path=pep_path)
            packed_pep = parse_pdb(path=pep_path)[0]
            x_pkl_file = os.path.join(save_path, pdb_id + f'_gen_{i}_X.pkl')
            pkl.dump(packed_pep['pos_heavyatom'].cpu().tolist(), open(x_pkl_file, 'wb')) # check here
            gen_lig_paths.append(pep_path)

        x_pkl_files.append(x_pkl_file) 

    metrics_otf = {}
    
    return batch, final_state, ref_merge, metrics_otf, pep_chain_id, receptor_chain_id, x_pkl_files, \
                gen_lig_paths, ref_lig_path, gen_pep_seqs

if __name__ == '__main__':
    pass