"""pep-rec dataset"""
import os
import logging
import joblib
import pickle
import lmdb
from Bio import PDB
from Bio.PDB import PDBExceptions
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from probayes.modules.common.geometry import *
from probayes.modules.protein.constants import *
from probayes.utils.data import PaddingCollate
from torch.utils.data import DataLoader


from torch.utils.data import DataLoader, Dataset

from probayes.utils.misc import load_config

import torch
import pandas as pd
from probayes.modules.protein.writers import save_pdb

# from remote.RDE_PPI.rde.utils.protein.parsers import parse_pdb
from probayes.data.parsers import parse_pdb
import os

os.chdir('/GenSIvePFS/wuhl/probayes')


class PepDataset(Dataset):

    MAP_SIZE = 32*(1024*1024*1024)  # 32GB

    def __init__(self, structure_dir = "./Data/PepMerge_new/", dataset_dir = "./Data/",
                                            name = 'pep', transform=None, reset=False, dataset_name='probayes'):
        super().__init__()
        self.structure_dir = structure_dir
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.dataset_name = dataset_name
        self.split_name = name
        self.split_indices = None
        
        self._load_indices(name)
        
        self.db_conn = None
        self.db_ids = None
        self._load_structures(reset)

    def _load_indices(self, mode):
        # load indices 
        structure_dir = os.path.dirname(self.structure_dir) # parent directory
        indices_fname = os.path.join(structure_dir, f'{mode}.txt')
        df = pd.read_csv(indices_fname, sep='\t',header=None)
        pdb_ids = list(df.iloc[:, 0].values)
        self.split_indices = pdb_ids

    @property
    def _cache_db_path(self):
        return os.path.join(self.dataset_dir, f'{self.split_name}_structure_cache.lmdb')

    def _connect_db(self):
        self._close_db()
        self.db_conn = lmdb.open(
            self._cache_db_path,
            map_size=self.MAP_SIZE,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db_conn.begin() as txn:
            keys = [k.decode() for k in txn.cursor().iternext(values=False)]
            self.db_ids = keys

    def _close_db(self):
        if self.db_conn is not None:
            self.db_conn.close()
        self.db_conn = None
        self.db_ids = None

    def _load_structures(self, reset):
        all_pdbs = os.listdir(self.structure_dir)

        if reset:
            if os.path.exists(self._cache_db_path):
                os.remove(self._cache_db_path)
                lock_file = self._cache_db_path + "-lock"
                if os.path.exists(lock_file):
                    os.remove(lock_file)
            self._close_db()
            todo_pdbs = all_pdbs
        else:
            if not os.path.exists(self._cache_db_path):
                todo_pdbs = all_pdbs
            else:
                todo_pdbs = []
                # self._connect_db()
                # processed_pdbs = self.db_ids
                # self._close_db()
                # todo_pdbs = list(set(all_pdbs) - set(processed_pdbs))

        if len(todo_pdbs) > 0:
            self._preprocess_structures(todo_pdbs)
    
    def _preprocess_structures(self, pdb_list):
        tasks = []
        for pdb_fname in pdb_list:
            pdb_path = os.path.join(self.structure_dir, pdb_fname)
            if os.path.exists(os.path.join(pdb_path,'pocket.pdb')) and\
                os.path.exists(os.path.join(pdb_path,'receptor.pdb')) and\
                    os.path.exists(os.path.join(pdb_path,'peptide.pdb')):
                tasks.append({
                    'id': pdb_fname,
                    'pdb_path': pdb_path,
                })

        data_list = joblib.Parallel(
            n_jobs = max(joblib.cpu_count() // 2, 1),
        )(
            joblib.delayed(self.preprocess_structure)(task)
            for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess')
        )
        # single cpu
        # data_list = [self.preprocess_structure(task) for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess')]

        db_conn = lmdb.open(
            self._cache_db_path,
            map_size = self.MAP_SIZE,
            create=True,
            subdir=False,
            readonly=False,
        )
        ids = []
        with db_conn.begin(write=True, buffers=True) as txn:
            for data in tqdm(data_list, dynamic_ncols=True, desc='Write to LMDB'):
                if data is None:
                    continue
                ids.append(data['id'])
                txn.put(data['id'].encode('utf-8'), pickle.dumps(data))

    def __len__(self):
        self._connect_db() # make sure db_ids is not None
        return len(self.db_ids)

    def __getitem__(self, index):
        self._connect_db()
        id = self.db_ids[index]
        with self.db_conn.begin() as txn:
            data = pickle.loads(txn.get(id.encode()))
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    def getitem_bypdbid(self, pdbid):

        pdb_path = os.path.join(self.structure_dir, str(pdbid))
        if os.path.exists(os.path.join(pdb_path,'pocket.pdb')) and\
            os.path.exists(os.path.join(pdb_path,'receptor.pdb')) and\
                os.path.exists(os.path.join(pdb_path,'peptide.pdb')):
            # pep
            pep = parse_pdb(os.path.join(pdb_path,'peptide.pdb'))[0]
            center = torch.sum(pep['pos_heavyatom'] [pep['mask_heavyatom'][:, BBHeavyAtom.CA], BBHeavyAtom.CA], dim=0) / (torch.sum(pep['mask_heavyatom'][:, BBHeavyAtom.CA]) + 1e-8)
            # rec
            rec = parse_pdb(os.path.join(pdb_path,'receptor.pdb'))[0]
            rec['pos_heavyatom'] = rec['pos_heavyatom'] - center[None, None, :]
            
            save_pdb(rec, os.path.join('cache_files/temp',f'{pdbid}_receptor_shifted.pdb'))
            print('save shifted recetor pdb to:', os.path.join('cache_files/temp',f'{pdbid}_receptor_shifted.pdb'))

    
    def preprocess_structure(self, task):
        # try:
        if task['id'] not in self.split_indices:
            # print(f'{task["id"]} not in names')
            return None
            # raise ValueError(f'{task["id"]} not in names')

        pdb_path = task['pdb_path']
        # pep
        # process peptide and find center of mass
        pep = parse_pdb(os.path.join(pdb_path,'peptide.pdb'))[0]
        if pep == None:
            print('pep parsing fault, pep is None!!!')
            return None
        
        center = torch.sum(pep['pos_heavyatom'] [pep['mask_heavyatom'][:, BBHeavyAtom.CA], BBHeavyAtom.CA], dim=0) / (torch.sum(pep['mask_heavyatom'][:, BBHeavyAtom.CA]) + 1e-8)
        pep['pos_heavyatom'] = pep['pos_heavyatom'] - center[None, None, :]
        
        pep['torsion_angle'] = torch.cat([pep['psi'].unsqueeze(-1),pep['chi']],dim=-1)
        pep['torsion_angle_mask'] = torch.cat([pep['psi_mask'].unsqueeze(-1),pep['chi_mask']],dim=-1)
        
        # if len(pep['aa'])<3 or len(pep['aa'])>25:
        #     print(len(pep['aa']))
            # raise ValueError('peptide length not in [3,25]')
        
        # rec
        rec = parse_pdb(os.path.join(pdb_path,'pocket.pdb'))[0]
        
        rec['pos_heavyatom'] = rec['pos_heavyatom'] - center[None, None, :]
        # rec['torsion_angle'],rec['torsion_angle_mask'] = get_torsion_angle(rec['pos_heavyatom'],rec['aa']) # calc angles after translation
        rec['torsion_angle'] = torch.cat([rec['psi'].unsqueeze(-1),rec['chi']],dim=-1)
        rec['torsion_angle_mask'] = torch.cat([rec['psi_mask'].unsqueeze(-1),rec['chi_mask']],dim=-1)
        
        
        rec['chain_nb'] += 1
        
        pep_chain_id = list(set(pep['chain_id']))
        assert len(pep_chain_id) == 1, 'pep_chain_id not the same'
        pep_chain_id = pep_chain_id[0]
        rec_chain_id = rec['chain_id']
        # rec_chain_ids = list(set(rec_chain_id))
        
        # meta data
        data = {}
        data['id'] = task['id']
        data['pep_chain_id'] = pep_chain_id
        # data['rec_chain_ids'] = rec_chain_ids
        data['generate_mask'] = torch.cat([torch.zeros_like(rec['aa']), torch.ones_like(pep['aa'])], dim=0).bool()
        
        for k in rec.keys():
            if isinstance(rec[k], torch.Tensor):
                data[k] = torch.cat([rec[k], pep[k]], dim=0)
            elif isinstance(rec[k], list):
                data[k] = rec[k] + pep[k]
            else:
                raise ValueError(f'Unknown type of {rec[k]}')
        return data
        



if __name__ == '__main__':
    # code to reset dataset cache
    device = 'cuda:1'
    config,cfg_name = load_config("configs/bfn_quat_pepbench.yaml")
    # config,cfg_name = load_config("configs/bfn_quat_pepbdb.yaml")
    # config,cfg_name = load_config("configs/bfn_prot_frag.yaml")
    # config,cfg_name = load_config("configs/learn_angle_testset.yaml")
    dataset = PepDataset(structure_dir = config.dataset.train.structure_dir, 
                         dataset_dir = config.dataset.train.dataset_dir,
                        name = config.dataset.train.name, 
                        dataset_name = config.dataset.name,
                        transform=None, reset=False)
    print(dataset[0]['generate_mask'].shape)

    val_dataset = PepDataset(structure_dir = config.dataset.val.structure_dir, dataset_dir = config.dataset.val.dataset_dir,
                                            name = config.dataset.val.name, transform=None, reset=False)
    dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=PaddingCollate(eight=False))

    test_dataset  = PepDataset(structure_dir = config.dataset.test.structure_dir, dataset_dir = config.dataset.test.dataset_dir,
                                            name = config.dataset.test.name, transform=None, reset=False)

    # from probayes.data import all_atom
    # from probayes.data import utils as du
    

    # batch = next(iter(dataloader))
    # print(batch.keys())
    # from openfold.np import residue_constants
    # from openfold.np.residue_constants import MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    # from probayes.modules.protein.constants import BBHeavyAtom
    # group_idx = torch.tensor(MAP_HHBLITS_AATYPE_TO_OUR_AATYPE)
    # rotmats_1 =  construct_3d_basis(batch['pos_heavyatom'][ :,:, BBHeavyAtom.CA],
    #                                 batch['pos_heavyatom'][ :,:,BBHeavyAtom.C],
    #                                 batch['pos_heavyatom'][ :,:,BBHeavyAtom.N] )
    # trans_1 = batch['pos_heavyatom'][ :,:,BBHeavyAtom.CA]
    # angles_1 = batch['torsion_angle']

    # gt_pos = batch['pos_heavyatom'][:,:,:-1,:]
    # cons_pos = all_atom.compute_allatom(
    #                     bb_rigids=du.create_rigid(rots=rotmats_1,trans=trans_1),
    #                     angles=angles_1,aatype=batch['aa'].clip(max=AA.UNK),)
    
    # res_mask = batch['mask_heavyatom'].gt(0)[:,:,:-1]
    # gt_pos_res = gt_pos[res_mask,...]
    # cons_pos_res = cons_pos[res_mask,...]
    # print(gt_pos.allclose(cons_pos))
    # print((gt_pos_res-cons_pos_res).abs().mean())
    
    test_dataset.getitem_bypdbid('3pkn')