import os
import logging
import joblib
import pickle
import lmdb
from Bio import PDB
from Bio.PDB import PDBExceptions
from torch.utils.data import Dataset
from tqdm.auto import tqdm

# from probayes.modules.protein.parsers import parse_pdb
from probayes.modules.common.geometry import *
from probayes.modules.protein.constants import *

from torch.utils.data import DataLoader, Dataset

from probayes.utils.misc import load_config

import torch
import pandas as pd
from probayes.modules.protein.writers import save_pdb
from remote.ppflow.tools.score.similarity import load_pep_seq

from remote.diffab.diffab.datasets.sabdab import preprocess_sabdab_structure,nan_to_empty_string,split_sabdab_delimited_str,nan_to_none,ALLOWED_AG_TYPES,RESOLUTION_THRESHOLD,parse_sabdab_resolution
import os
import numpy as np
from anarci.anarci import get_identity
# os.chdir('/data/wuhl/bfn4pep')


class AntibodyDataset(Dataset):

    MAP_SIZE = 32*(1024*1024*1024)  # 32GB

    def __init__(self, structure_dir = "./Data/PepMerge_new/", dataset_dir = "./Data/",
                                            name = 'pep', transform=None, reset=False, split_json_path =None, dataset_name='probayes', n_cpu=8, numbering_scheme='chothia'):
        super().__init__()
        self.structure_dir = structure_dir
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.dataset_name = dataset_name
        # self.test_seqs = self._load_test_seqs()
        self.summary_path = '/data/wuhl/bfn4pep/probayes_data/antibody_processed/sabdab_summary.tsv'

        self.pdbid2entry_idx = {}
        self.sabdab_entries = None        
        self.split_name = name
        self.split_indices = None
        self.split_json_path = split_json_path
        self._load_indices(name)

        self.entryid2entry = {}
        self._load_entries()


        self.n_cpu = n_cpu
        self.numbering_scheme = numbering_scheme
        self.db_conn = None
        self.db_ids = None
        self._load_structures(reset)
        
    def _load_entries(self):
        # df = pd.read_csv(self.summary_path, sep='\t')
        entries_all = []
        
        for i, row in tqdm(
            self.df.iterrows(), 
            dynamic_ncols=True, 
            desc='Loading entries',
            total=len(self.df), disable=True
        ):       
            
            entry_id = "{pdbcode}_{H}_{L}_{Ag}".format(
                pdbcode= row['pdb'],
                H = row['heavy_chain'],
                L = row['light_chain'],
                Ag = ''.join(
                    row['antigen_chains']
                )
            )
            entry = {
                'id': entry_id,
                'pdbcode': row['pdb'],
                'H_chain': nan_to_none(row['heavy_chain']),
                'L_chain': nan_to_none(row['light_chain']),
                'ag_chains': row['antigen_chains'],
                'pdb_path': row['pdb_data_path']
            }
            entries_all.append(entry)
            # self.pdbid2entry[row['pdb']] = entry
            self.entryid2entry[entry_id] = entry
        

        self.entries = entries_all

    def _load_test_seqs(self):
        test_json_path = '/data/wuhl/bfn4pep/probayes_data/antibody_processed/test.json'
        test_df = pd.read_json(test_json_path, lines=True)

        test_seqs = []
        
        for i, row in tqdm(
            test_df.iterrows(), 
            dynamic_ncols=True, 
            desc='Loading Testset Seqs',
            total=len(test_df), disable=False
        ):
            pdb_path = row['pdb_data_path']
            test_seqs.append(load_pep_seq(pdb_path))

        return test_seqs

    def _load_indices(self, mode):
        # load indices 
        self.df = pd.read_json(self.split_json_path, lines=True)
        pdb_ids = list(self.df['pdb'].values)
        self.items = list(self.df.itertuples())
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

        if len(todo_pdbs) > 0:
            self._preprocess_structures(todo_pdbs)
    
    def _preprocess_structures(self, pdb_list):
        tasks = []
        for entry in self.entries:
            tasks.append({
                'id': entry['pdbcode'],
                'entry': entry,
                'pdb_path': entry['pdb_path'],
            })

        if self.n_cpu != 1:
            data_list = joblib.Parallel(
                n_jobs = max(joblib.cpu_count() // 2, 1),
            )(
                joblib.delayed(preprocess_sabdab_structure)(task)
                for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess')
            )
        else:
            # single cpu
            data_list = [preprocess_sabdab_structure(task) for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess')]

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

    def __getitem__(self, index, do_transform=True):
        self._connect_db()
        id = self.db_ids[index%len(self.db_ids)]
        with self.db_conn.begin() as txn:
            data = pickle.loads(txn.get(id.encode()))
        if self.transform is not None and do_transform:
            data = self.transform(data)
        data['id'] = id
        return data
    
    def get_structure_bypdbid(self, pdb_id, do_transform=False):
        # entry = self.entryid2entry[pdb_id]
        # task = {
        #         'id': entry['pdbcode'],
        #         'entry': entry,
        #         'pdb_path': entry['pdb_path'],
        #     }
        # structure = preprocess_sabdab_structure(task)
        # structure['id'] = pdb_id
        self._connect_db()
        id = pdb_id
        with self.db_conn.begin() as txn:
            data = pickle.loads(txn.get(id.encode()))
        if self.transform is not None and do_transform:
            data = self.transform(data)
        data['id'] = id        
        return data
    
        
if __name__ == '__main__':
    from remote.diffab.diffab.utils.transforms import get_transform

    # code to reset dataset cache
    device = 'cuda:1'
    reset = False
    n_cpu = 32
    config,cfg_name = load_config("configs/bfn_antibody.yaml")

    test_dataset  = AntibodyDataset(structure_dir = config.dataset.test.structure_dir, dataset_dir = config.dataset.test.dataset_dir, split_json_path= config.dataset.test.split_json_path,
                                            name = config.dataset.test.name, transform=get_transform(config.dataset.test.transform), reset=reset, n_cpu=n_cpu)
    # test_seq_lens = [len(dataset[i]['aa']) for i in range(len(test_dataset))]
    
    # config,cfg_name = load_config("configs/learn_angle_testset.yaml")
    dataset = AntibodyDataset(structure_dir = config.dataset.train.structure_dir, 
                         dataset_dir = config.dataset.train.dataset_dir,
                        name = config.dataset.train.name, 
                        dataset_name = config.dataset.name,
                        split_json_path = config.dataset.train.split_json_path,
                        transform=get_transform(config.dataset.train.transform), 
                        reset=reset, 
                        n_cpu=n_cpu,
                        numbering_scheme=config.dataset.numbering_scheme,
                        )
    print(len(dataset))
    print(dataset[0].keys())
    # seq_lens = [len(dataset[i]['aa']) for i in range(len(dataset))]

    # val_dataset = AntibodyDataset(structure_dir = config.dataset.val.structure_dir, dataset_dir = config.dataset.val.dataset_dir,split_json_path = config.dataset.val.split_json_path,
    #                                         name = config.dataset.val.name, transform=get_transform(config.dataset.val.transform), reset=reset, n_cpu=n_cpu)
    