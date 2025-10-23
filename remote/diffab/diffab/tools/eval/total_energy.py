# calculate the total energy
import pyrosetta
from pyrosetta import *
pyrosetta.init(' '.join([
    '-mute', 'all',
    '-use_input_sc',
    '-ignore_unrecognized_res',
    '-ignore_zero_occupancy', 'false',
    '-load_PDB_components', 'false',
    '-relax:default_repeats', '2',
    '-no_fconfig',
    '-constant_seed'
]))

from remote.diffab.diffab.tools.eval.base import EvalTask
from pyrosetta.rosetta.core.simple_metrics.per_residue_metrics import PerResidueEnergyMetric
from pyrosetta.rosetta.core.select import residue_selector as selections


def pyrosetta_total_energy(task: EvalTask):
    pose = pose_from_pdb(task.in_path)
    # ref_pose = pose_from_pdb(task.ref_path)
    ref_pose = pose_from_pdb(task.relaxed_ref_path)
    # calculate the total energy
    # scorefxn = pyrosetta.get_score_function() 
    # scorefxn = create_score_function('ref2015')
    scorefxn = get_fa_scorefxn()
    metric = PerResidueEnergyMetric()
    metric.set_scorefunction(scorefxn)
    
    per_residue_energies = metric.calculate(pose)
    ref_per_residue_energies = metric.calculate(ref_pose)
    
    # select the cdrh3 region
    assert task.residue_first[0] == task.residue_last[0], 'cdrh3 region should be in the same chain'
    heavy_chain_id = task.residue_first[0]
    cdr_pose_indices = [pose.pdb_info().pdb2pose(heavy_chain_id, resseq, icode) for (_, resseq, icode) in task.cdrh3_reslist]
    task.cdrh3_seq = ''.join([ref_pose.residue(i).name1() for i in cdr_pose_indices])
    
    # calculate energies
    cdr_energy = sum([per_residue_energies[i] for i in cdr_pose_indices])
    ref_cdr_energy = sum([ref_per_residue_energies[i] for i in cdr_pose_indices]) 
   
    task.scores.update({
        'gen_total_energy_sum': cdr_energy,
        'ref_total_energy_sum': ref_cdr_energy,
    })
    return task


def eval_total_energy(task: EvalTask):
    task = pyrosetta_total_energy(task)
    return task

if __name__ == '__main__':
    from remote.PepGLAD.data.converter.pdb_to_list_blocks import pdb_to_list_blocks
    from remote.PepGLAD.data.converter.list_blocks_to_pdb import list_blocks_to_pdb

    from pyrosetta.rosetta.protocols.antibody import AntibodyInfo, CDRDefinitionEnum
    from pyrosetta.rosetta.protocols.antibody.residue_selector import CDRResidueSelector
    from pyrosetta.rosetta.protocols.antibody import CDRNameEnum
    from pyrosetta import *
    # "cdrh3_pos": [96, 110]
    
    # task = EvalTask(
    #     in_path=pdb_path,
    #     ref_path=pdb_path,
    #     info=None,
    #     structure=None,
    #     name=None,
    #     method=None,
    #     cdr='cdrh3',
        
    # )
    
    pdb_path = '/data/wuhl/bfn4pep/logs/bfn_antibody[dev-4b14445][05-11-01-25-09]_no_seq_sc_mask_mixsc/results/candidates/1a14_H_L_N_gen_rosetta.pdb'

    pose = pose_from_pdb(pdb_path)
    
    # ab_info = AntibodyInfo(pose, CDRDefinitionEnum.North_Dunbrack)

    ab_info = AntibodyInfo(pose, CDRDefinitionEnum.Chothia)

    # scoring
    scorefxn = pyrosetta.get_score_function()  # 默认是 'ref15'

    # scorefxn = create_score_function('ref15')
    # h_chain
    cdr_pose_indices = [pose.pdb_info().pdb2pose('H', 105) ]

    metric = PerResidueEnergyMetric()
    # scorefxn = create_score_function('ref2015')
    metric.set_scorefunction(scorefxn)
    per_residue_energies = metric.calculate(pose)
    print(per_residue_energies)
    