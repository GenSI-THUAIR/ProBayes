import numpy as np
from probayes.core.flow_model import FlowModel
from tqdm import tqdm
import gc
import torch
import json
import tempfile
from probayes.utils.train import recursive_to
from probayes.dataset.pep_dataset import PepDataset
from probayes.core.sample import sample_from_data, generate
from probayes.eval.geometry import get_rmsd, align_chains, diff_ratio
from Bio.SeqUtils import seq1
from Bio.PDB import PDBExceptions, PDBIO
from Bio.PDB import PDBParser, Superimposer, is_aa, Select, NeighborSearch
import mdtraj as md
import os.path as osp
from mdtraj.core.trajectory import Trajectory
from probayes.eval.sequence_sim import ESM_Helper, calculate_fid
import os
from multiprocessing import Pool
import re
from remote.ppflow.tools.score.similarity import tm_align_score, seq_similarity, tm_score
from remote.ppflow.tools.score import similarity, check_validity, foldx_energy
from tqdm import trange
from remote.PepGLAD.generate import save_data
from remote.ppflow.tools.score.similarity import load_pep_seq
import pickle as pkl
from probayes.eval.geometry import so3_wasserstein
# from remote.RDE_PPI.rde.utils.protein.parsers import parse_pdb

from probayes.data.parsers import parse_pdb

def get_chain_from_structure(structure, chain_id='A'):
    for chain in structure:
        if chain.id == chain_id:
            # print(len(chain))
            return chain
    raise ValueError(f'Chain {chain_id} not found in structure')

def get_bind_site(pep_chain, complex_chains, pep_chain_id):
    # parser = PDBParser()
    # structure = parser.get_structure('X', pdb)[0]
    
    peps = [atom for res in pep_chain for atom in res if atom.get_name() == 'CA']
    recs = [atom for chain in complex_chains
            if chain.get_id()!=pep_chain_id for res in chain for atom in res if atom.get_name() == 'CA']
    # print(recs)
    search = NeighborSearch(recs)
    near_res = []
    for atom in peps:
        near_res += search.search(atom.get_coord(), 10.0, level='R')
    near_res = set([res.get_id()[1] for res in near_res])
    return near_res

def get_bind_ratio(pdb1, pdb2, chain_id1, chain_id2):
    near_res1,near_res2 = get_bind_site(pdb1,chain_id1),get_bind_site(pdb2,chain_id2)
    return len(near_res1.intersection(near_res2))/(len(near_res2)+1e-10) # last one is gt

def get_second_stru(pdb,chain):
    parser = PDBParser()
    structure = parser.get_structure('X', pdb)[0]
    chain2id = {chain.id:i for i,chain in enumerate(structure)}
    traj = md.load(pdb)
    chain_indices = traj.topology.select(f"chainid {chain2id[chain]}")
    traj = traj.atom_slice(chain_indices)
    return md.compute_dssp(traj,simplified=True)

def get_ss(traj1,traj2):
    # traj1,traj2 = get_traj_chain(pdb1,chain_id1),get_traj_chain(pdb2,chain_id2)
    ss1,ss2 = md.compute_dssp(traj1,simplified=True),md.compute_dssp(traj2,simplified=True)
    return (ss1==ss2).mean()

def dockq(mod_pdb: str, native_pdb: str, pep_chain: str, DOCKQ_DIR:str):
    p = os.popen(f'{os.path.join(DOCKQ_DIR, "DockQ.py")} {mod_pdb} {native_pdb} -model_chain1 {pep_chain} -native_chain1 {pep_chain} -no_needle')
    text = p.read()
    p.close()
    res = re.search(r'DockQ\s+([0-1]\.[0-9]+)', text)
    score = float(res.group(1))
    return score

def structure2temppdb(structure):
    temp_pdb_file = tempfile.NamedTemporaryFile(suffix='.pdb')
    temp_pdb_path = temp_pdb_file.name
    io = PDBIO()
    io.set_structure(structure)
    io.save(temp_pdb_path)
    return temp_pdb_path, temp_pdb_file

# def structure2temppdb(structure):
#     temp_pdb_file = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)
#     temp_pdb_path = temp_pdb_file.name
#     temp_pdb_file.close()  # Close the file so that other processes can access it

#     # Set the file permissions to be readable and writable by the owner
#     os.chmod(temp_pdb_path, 0o777)

#     io = PDBIO()
#     io.set_structure(structure)
#     io.save(temp_pdb_path)
#     return temp_pdb_path, temp_pdb_file

def get_metrics(model:FlowModel, val_dataset:PepDataset, n_steps=200, sample_bb=True, sample_ang=True, sample_seq=True, device='cuda',n_samples=64):
    model.eval()
    esm_helper = ESM_Helper(device=device)
    aars, rmsds, BSRs, SSRs = [], [], [], []
    mae_chi1, mae_chi2, mae_chi3, mae_chi4,corr_rate = [], [], [], [], []
    ref_seqs_embeds, gen_seqs_embeds = [], []
    dockq_scores = []
    for i, data in enumerate(tqdm(val_dataset, desc='Get Metrics', dynamic_ncols=True)):
        try:
            sample_pep_structures, ref_pep_structure, sample_merges, ref_merge, angle_metrics = \
                sample_from_data(data=data,model=model,device=device,num_samples=n_samples, 
                                 num_steps=n_steps,sample_bb=sample_bb,sample_ang=sample_ang,sample_seq=sample_seq)
        except (PDBExceptions.PDBConstructionException, torch.cuda.OutOfMemoryError) as e:
            print(f'building id:{data["id"]} error, here is the info', e)
            continue
        mae_chi1.append(angle_metrics['mae_chi1'])
        mae_chi2.append(angle_metrics['mae_chi2'])
        mae_chi3.append(angle_metrics['mae_chi3'])
        mae_chi4.append(angle_metrics['mae_chi4'])
        corr_rate.append(angle_metrics['correct_rate'])
        
        # get sample and reference peptide structure
        sample_pep_chains = [list(s.get_chains()) for s in sample_pep_structures]
        ref_pep_chains = list(ref_pep_structure.get_chains())
        assert len(ref_pep_chains) == 1\
            and np.prod([len(ch) for ch in sample_pep_chains]) == 1,\
            'Only support single peptide chain'
        
        # get sample and reference peptide chain
        ref_pep_ch = ref_pep_chains[0]
        sample_pep_chains = [ch[0] for ch in sample_pep_chains]
        
        # get the peptide chain id
        pep_chain_id = ref_pep_ch.id
        
        # reference bind site
        ref_near_res = get_bind_site(ref_pep_ch, list(ref_merge.get_chains()), pep_chain_id)
        
        # reference second structure
        ref_temppdb_path, ref_temppdb_file = structure2temppdb(ref_merge)
        ref_second_structure = get_second_stru(ref_temppdb_path, pep_chain_id)

        for sample_pep_id, sample_pep_ch in enumerate(sample_pep_chains):
            # aar
            ref_res, sample_res = align_chains(ref_pep_ch, sample_pep_ch)
            ref_seq = seq1("".join([residue.get_resname() for residue in ref_res]))
            ref_seqs_embeds.extend(esm_helper.seqs_embedding(ref_seq))
            sample_seq = seq1("".join([residue.get_resname() for residue in sample_res]))
            gen_seqs_embeds.extend(esm_helper.seqs_embedding(sample_seq))
            aar = diff_ratio(ref_seq, sample_seq)
            aars.append(aar)
            # RMSD
            _, rmsd = get_rmsd(ref_pep_ch, sample_pep_ch)
            rmsds.append(rmsd)
            # Bind Site Ratio
            sample_res = get_bind_site(sample_pep_ch, list(ref_merge.get_chains()), pep_chain_id)
            BSRs.append(len(sample_res.intersection(ref_near_res))/(len(ref_near_res)+1e-10)) # last one is gt
            # Second Structure Ratio
            sample_temppdb_path, _ = structure2temppdb(sample_merges[sample_pep_id])
            sample_second_structure = get_second_stru(sample_temppdb_path, pep_chain_id)
            SSRs.append((sample_second_structure==ref_second_structure).mean())
    
    data_feature = torch.stack(ref_seqs_embeds)
    gen_feature = torch.stack(gen_seqs_embeds)
    fid = calculate_fid(data_feature,gen_feature)
    return {
        'AAR': np.mean(aars),
        'RMSD': np.mean(rmsds),
        'BSR': np.mean(BSRs),
        'SSR': np.mean(SSRs),
        'mae_chi1': float(torch.cat(mae_chi1,dim=0).mean().cpu()),
        'mae_chi2': float(torch.cat(mae_chi2,dim=0).mean().cpu()),
        'mae_chi3': float(torch.cat(mae_chi3,dim=0).mean().cpu()),
        'mae_chi4': float(torch.cat(mae_chi4,dim=0).mean().cpu()),
        'correct_rate': np.mean(corr_rate),
        'fid': fid,
    }


def save_pickle(data,path):
    pkl_file = os.path.join(path)
    pkl.dump(data, open(pkl_file, 'wb'))
    


class Evaluator:
    def __init__(self):
        self.ref_states = []
        self.final_states = []
        self.ref_pep_paths = []
        self.gen_pep_paths = []
    
    def add_pair(self, ref_state, final_state):
        self.ref_states.append(ref_state)
        self.final_states.append(final_state)
    
    def add_pair_path(self, ref_pep_path, gen_pep_path):
        self.ref_pep_paths.append(ref_pep_path)
        self.gen_pep_paths.append(gen_pep_path)

    def get_so3_wdist(self, wdist_size=5000):
        if len(self.ref_states) == 0:
            return {}
        
        ref_rotmats = []
        gen_rotmats = []
        for i in range(len(self.ref_states)):
            gen_mask = self.ref_states[i]['generate_mask']
            mask = gen_mask.gt(0)
            ref_rotmats.append(self.final_states[i]['rotmats_1'][mask,:])
            gen_rotmats.append(self.final_states[i]['rotmats'][mask,:])
        ref_rotmats = torch.cat(ref_rotmats, dim=0)
        gen_rotmats = torch.cat(gen_rotmats, dim=0)

        random_index = torch.randperm(ref_rotmats.size(0))[:wdist_size]
        ref_rotmats = ref_rotmats[random_index]
        gen_rotmats = gen_rotmats[random_index]
        
        wdist = so3_wasserstein(ref_rotmats, gen_rotmats)
        return {'SO3_wdist': wdist}
    
    def get_psi_mae(self):
        ref_psi = []
        gen_psi = []
        maes = []
        for id, ref_pep_path in enumerate(self.ref_pep_paths):
            ref_psi = parse_pdb(path=ref_pep_path)[0]['psi']
            psi_mae_this_id = []
            for gen_pep_path in self.gen_pep_paths[id]:
                gen_psi = parse_pdb(path=gen_pep_path)[0]['psi']
                psi_mae = ((ref_psi-gen_psi)%(2*np.pi)).abs().mean()
                psi_mae_this_id.append(psi_mae)
            maes.append(torch.mean(torch.tensor(psi_mae_this_id)))
        psi_mae_avg = torch.tensor(maes).mean()
        # wdist = so3_wasserstein(ref_rotmats, gen_rotmats)
        return {'psi_mae': float(psi_mae_avg.cpu())}

    def get_diversity(self):
        seq_diversity = []
        struct_diversity = []
        diversity = []
        for i in trange(len(self.ref_pep_paths), desc='Get Diversity', dynamic_ncols=True):
            # ref_pep_path = self.ref_pep_paths[i]
            gen_pep_paths = self.gen_pep_paths[i]
            
            this_seq_diversity = []
            this_struct_diversity = []
            this_diversity = []
            pep_seq = load_pep_seq(gen_pep_paths[0])
            if len(pep_seq) <= 3:
                print(f'Peptide sequence is too short: {pep_seq}')
                continue #TM score will raise error, skip this
            
            indices = np.tril_indices(len(gen_pep_paths), -1)
            for id1, id2 in zip(*indices):
                try:
                    tm = tm_score(gen_pep_paths[id1], gen_pep_paths[id2])
                except Exception as e:
                    print(f'Error in tm_score: {e}')
                    continue
                this_struct_diversity.append(1-tm)
                seq_sim = seq_similarity(gen_pep_paths[id1], gen_pep_paths[id2])
                this_seq_diversity.append(1-seq_sim)
                this_diversity.append((1-tm)*(1-seq_sim))
            
            seq_diversity.append(np.mean(this_seq_diversity))
            struct_diversity.append(np.mean(this_struct_diversity))
            diversity.append(np.mean(this_diversity))
            
        return {
            'seq_diversity_OL': np.mean(seq_diversity),
            'struct_diversity_TM': np.mean(struct_diversity),
            'diversity_TMSeqOL': np.mean(diversity),
        }
    
    def get_validity(self):
        validity = []
        for i in trange(len(self.ref_pep_paths), desc='Get Validity', dynamic_ncols=True):
            # ref_pep_path = self.ref_pep_paths[i]
            gen_pep_paths = self.gen_pep_paths[i]
            this_validity = []
            for gen_pep_path in gen_pep_paths:
                valid = check_validity.bond_length_validation(gen_pep_path)
                this_validity.append(valid)
            validity.append(np.mean(this_validity))
        return {'validity': np.mean(validity)}
    
    def get_novelty(self):
        novelty = []
        for i in trange(len(self.ref_pep_paths), desc='Get Novelty', dynamic_ncols=True):
            ref_pep_path = self.ref_pep_paths[i]
            gen_pep_paths = self.gen_pep_paths[i]
            this_novelty = []
            pep_seq = load_pep_seq(ref_pep_path)
            if len(pep_seq) <= 3:
                print(f'Peptide sequence is too short: {pep_seq}')
                continue #TM score will raise error, skip this
            for gen_pep_path in gen_pep_paths:
                seq_novel = similarity.seq_similarity(ref_pep_path, gen_pep_path) < 0.5
                try:
                    struct_novel = tm_score(ref_pep_path, gen_pep_path) < 0.5
                except Exception as e:
                    print(f'Error in tm_score: {e}')
                    continue
                is_novel = seq_novel and struct_novel
                this_novelty.append(is_novel)
            novelty.append(np.mean(this_novelty))
        return {'novelty': np.mean(novelty)}
    
    def get_metrics(self):
        metrics = {}
        # metrics.update(self.get_so3_wdist())
        # metrics.update(self.get_psi_mae())
        metrics.update(self.get_diversity())
        metrics.update(self.get_validity())
        metrics.update(self.get_novelty())
        return metrics

def prepare_result_json(ckpt_path:str, model:FlowModel, val_dataset:PepDataset,
                          n_steps=200, sample_bb=True, sample_ang=True,
                          sample_seq=True, device='cuda',n_samples=64, dataset_pdb_dir=None, sample_mode=None,sc_pack='rosetta'):
    
    gen_eval = Evaluator()
    
    model.eval()
    if not os.path.exists(os.path.join(ckpt_path,'results')):
        os.mkdir(os.path.join(ckpt_path,'results'))
    json_path = os.path.join(ckpt_path, 'results/' 'results.jsonl')
    fout = open(json_path, 'w')
    
    for i, data in enumerate(tqdm(val_dataset, desc='Get Metrics', dynamic_ncols=True)):
        ref_state, final_state, \
            ref_complex_path,\
                metrics_otf, pep_chain_id, receptor_chain_id,\
                    x_pkl_file, \
                    gen_lig_path, ref_lig_path, gen_pep_seqs = generate(ckpt_path, 
                            data=data,model=model,device=device,num_samples=n_samples, 
                                num_steps=n_steps,sample_bb=sample_bb,sample_ang=sample_ang,sample_seq=sample_seq, 
                                dataset_pdb_dir=dataset_pdb_dir,sample_mode=sample_mode,sc_pack=sc_pack)
        
        gen_eval.add_pair(ref_state, final_state)
        gen_eval.add_pair_path(ref_pep_path=ref_lig_path,gen_pep_path=gen_lig_path)
        inputs = []
        
        # process each (e.g. 64) sample peptide
        for sample_pep_id, sample_pep_seq in enumerate(gen_pep_seqs):
            if os.path.exists(x_pkl_file[sample_pep_id]):
                inputs.append(
                        (
                        data['id'],sample_pep_id,
                        x_pkl_file[sample_pep_id], sample_pep_seq,
                        ref_complex_path, receptor_chain_id,pep_chain_id,
                        os.path.join(ckpt_path,'results','references'),
                        os.path.join(ckpt_path,'results','candidates'),
                        ref_lig_path,gen_lig_path[sample_pep_id],
                        (sample_seq and (not sample_bb) and (not sample_ang)), # seq_only, 
                        ((not sample_seq) and sample_bb and sample_ang), # struct_only
                        ((not sample_seq) and sample_bb and (not sample_ang)), # backbone_only
                        ))
            else:
                raise ValueError(f'No pkl file found for {x_pkl_file[sample_pep_id]}')
            
        results = []
        for inp in inputs:
            results.append(save_data(*inp))
        for result in results:
            fout.write(json.dumps(result) + '\n')
    fout.close()
    
    metrics = gen_eval.get_metrics()
    metrics.update(metrics_otf)
    return json_path, metrics






