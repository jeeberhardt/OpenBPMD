#!/usr/bin/env python
descriptn = \
    """
    OpenBPMD - an open source implementation of Binding Pose Metadynamics
    (BPMD) with OpenMM. Replicates the protocol as described by
    Clark et al. 2016 (DOI: 10.1021/acs.jctc.6b00201).

    Runs ten 10 ns metadynamics simulations that biases the RMSD of the ligand.

    The stability of the ligand is calculated using the ligand RMSD (PoseScore)
    and the persistence of the original noncovalent interactions between the
    protein and the ligand (ContactScore). Stable poses have a low RMSD and
    a high fraction of the native contacts preserved until the end of the
    simulation.

    A composite score is calculated using the following formula:
    CompScore = PoseScore - 5 * ContactScore
    """

# OpenMM
from openmm.app import *
from openmm import *

# The rest
import argparse
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms, contacts
import mdtraj as md
import pandas as pd
import parmed as pmd
import glob
import os

__author__ = "Dominykas Lukauskis"
__version__ = "1.0.0"
__email__ = "dominykas.lukauskis.19@ucl.ac.uk"
    

def get_contact_score(structure_file, trajectory_file, lig_resname):
    """A function the gets the ContactScore from an OpenBPMD trajectory.

    Parameters
    ----------
    structure_file : str
        The name of the centred equilibrated system PDB file that 
        was used to start the OpenBPMD simulation.
    trajectory_file : str
        The name of the OpenBPMD trajectory file.
    lig_resname : str
        Residue name of the ligand that was biased.

    Returns
    -------
    contact_scores : np.array 
        ContactScore for every frame of the trajectory.
    """
    u = mda.Universe(structure_file, trajectory_file)

    sel_donor = f"resname {lig_resname} and not name *H*"
    sel_acceptor = f"protein and not name H* and \
                     around 5 resname {lig_resname}"

    # reference groups (first frame of the trajectory, but you could also use
    # a separate PDB, eg crystal structure)
    a_donors = u.select_atoms(sel_donor)
    a_acceptors = u.select_atoms(sel_acceptor)

    cont_analysis = contacts.Contacts(u, select=(sel_donor, sel_acceptor),
                                      refgroup=(a_donors, a_acceptors),
                                      radius=3.5)

    cont_analysis.run()
    # print number of average contacts in the first ns
    # NOTE - hard coded number of frames (100 per traj)
    frame_idx_first_ns = int(len(cont_analysis.results.timeseries)/10)
    first_ns_mean = np.mean(cont_analysis.results.timeseries[1:frame_idx_first_ns, 1])
    if first_ns_mean == 0:
        normed_contacts = cont_analysis.results.timeseries[1:, 1]
    else:
        normed_contacts = cont_analysis.results.timeseries[1:, 1]/first_ns_mean
    contact_scores = np.where(normed_contacts > 1, 1, normed_contacts)

    return contact_scores


def get_pose_score(structure_file, trajectory_file, lig_resname):
    """A function the gets the PoseScore (ligand RMSD) from an OpenBPMD
    trajectory.

    Parameters
    ----------
    'structure_file : str
        The name of the centred equilibrated system
        PDB file that was used to start the OpenBPMD simulation.
    trajectory_file : str
        The name of the OpenBPMD trajectory file.
    lig_resname : str
        Residue name of the ligand that was biased.

    Returns
    -------
    pose_scores : np.array 
        PoseScore for every frame of the trajectory.
    """
    # Load a MDA universe with the trajectory
    u = mda.Universe(structure_file, trajectory_file)
    # Align each frame using the backbone as reference
    # Calculate the RMSD of ligand heavy atoms
    r = rms.RMSD(u, select='backbone',
                 groupselections=[f'resname {lig_resname} and not name H*'],
                 ref_frame=0).run()
    # Get the PoseScores as np.array
    pose_scores = r.results.rmsd[1:, -1]

    return pose_scores


def add_harmonic_restraints(prmtop, inpcrd, system, atom_selection, k=1.0):
    """Add harmonic restraints to the system.

    Args:
        prmtop (AmberPrmtopFile): Amber topology object
        inpcrd (AmberInpcrdFile): Amber coordinates object
        system (System): OpenMM system object created from the Amber topology object
        atom_selection (str): Atom selection (see MDTraj documentation)
        k (float): harmonic force restraints in kcal/mol/A**2 (default: 1 kcal/mol/A**2)

    Returns:
        (list, int): list of all the atom ids on which am harmonic force is applied, index with the System of the force that was added

    """
    mdtop = md.Topology.from_openmm(prmtop.topology)
    atom_idxs = mdtop.select(atom_selection)
    positions = inpcrd.positions

    # Tranform constant to the right unit
    k = k.value_in_unit_system(unit.md_unit_system)

    if atom_idxs.size == 0:
        print("Warning: no atoms selected using: %s" % atom_selection)
        return ([], None)

    # Take into accoun the periodic condition
    # http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomExternalForce.html
    force = CustomExternalForce("k * periodicdistance(x, y, z, x0, y0, z0)^2")
    
    harmonic_force_idx = system.addForce(force)
    
    force.addGlobalParameter("k", k)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    
    for atom_idx in atom_idxs:
        #print(atom_idx, positions[atom_idx].value_in_unit_system(units.md_unit_system))
        force.addParticle(int(atom_idx), positions[atom_idx].value_in_unit_system(unit.md_unit_system))

    return atom_idxs


def minimisation(parm, coords, min_file_name):
    """An energy minimization function down with an energy tolerance
    of 10 kJ/mol.

    Parameters
    ----------
    parm : Parmed or OpenMM parameter file object
        Used to create the OpenMM System object.
    input_positions : OpenMM Quantity
        3D coordinates of the equilibrated system.
    min_file_name : str
        Name of the minimized PDB file to write.
    """
    # Configuration system
    system = parm.createSystem(nonbondedMethod=PME, nonbondedCutoff=12 * unit.angstrom, constraints=HBonds)

    properties = {"Precision": "mixed"}
    platform = Platform.getPlatformByName('OpenCL')
    integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)
    simulation = Simulation(parm.topology, system, integrator, platform, properties)
    simulation.context.setPositions(coords.positions)

    print('Energy minimization')
    simulation.minimizeEnergy()

    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    with open(min_file_name, 'w') as output:
        output.write(XmlSerializer.serialize(state))


def equilibrate(parm, coords, min_file_name, eq_file_name, lig_resname):
    """A function that does a 500 ps NPT equilibration with position
    restraints, with a 5 kcal/mol/A**2 harmonic constant on solute heavy
    atoms, using a 2 fs timestep.

    Parameters
    ----------
    min_pdb : str
        Name of the minimized PDB file.
    parm : Parmed or OpenMM parameter file object
        Used to create the OpenMM System object.
    out_dir : str
        Directory to write the outputs to.
    eq_file_name : str
        Name of the equilibrated PDB file to write.
    """
    steps = 250000
    restraint_value = 2.5 * unit.kilocalories_per_mole / unit.angstroms**2

    # Configuration system
    system = parm.createSystem(nonbondedMethod=PME, nonbondedCutoff=12 * unit.angstrom, constraints=HBonds)

    atom_idxs = add_harmonic_restraints(parm, coords, system, "(protein or resname %s) and not element H" % lig_resname, restraint_value)
    print('Number of particles constrainted: %d' % len(atom_idxs))

    print('Create simulation system')
    properties = {"Precision": "mixed"}
    platform = Platform.getPlatformByName('OpenCL')
    system.addForce(MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin))
    integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)
    simulation = Simulation(parm.topology, system, integrator, platform, properties)
    simulation.context.setPositions(coords.positions)

    simulation.loadState(min_file_name)

    print('Equilibration with constraint value of %s' % restraint_value)
    # MD simulations - production
    simulation.reporters.append(DCDReporter('equil.dcd', 500)) # 1 ps
    simulation.reporters.append(CheckpointReporter('equil.chk', 50000)) # 100 ps
    simulation.reporters.append(StateDataReporter('equil.log', 500, 
                                step=True, temperature=True, progress=True, time=True, 
                                potentialEnergy=True, kineticEnergy=True, totalEnergy=True, 
                                remainingTime=True, speed=True, volume=True, density=True,
                                totalSteps=steps, separator=',')) # 1 ps

    simulation.step(steps) # 500 ps

    state = simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
    with open(eq_file_name, 'w') as output:
        output.write(XmlSerializer.serialize(state))


def produce(parm, coords, lig_resname, anchor_residues=None, set_hill_height=0.3, rep_dir='.', eq_file_name='equil.xml'):
    """An OpenBPMD production simulation function. Ligand RMSD is biased with
    metadynamics. The integrator uses a 4 fs time step and
    runs for 10 ns, writing a frame every 100 ps.

    Writes a 'trj.dcd', 'COLVAR.npy', 'bias_*.npy' and 'sim_log.csv' files
    during the metadynamics simulation in the '{out_dir}/rep_{idx}' directory.
    After the simulation is done, it analyses the trajectories and writes a
    'bpm_results.csv' file with time-resolved PoseScore and ContactScore.

    Parameters
    ----------
    out_dir : str
        Directory where your equilibration PDBs and 'rep_*' dirs are at.
    idx : int
        Current replica index.
    lig_resname : str
        Residue name of the ligand.
    eq_pdb : str
        Name of the PDB for equilibrated system.
    parm : Parmed or OpenMM parameter file object
        Used to create the OpenMM System object.
    parm_file : str
        The name of the parameter or topology file of the system.
    coords_file : str
        The name of the coordinate file of the system.
    set_hill_height : float
        Metadynamic hill height, in kcal/mol.

    """
    sim_time = 10  # ns
    steps = 250000 * sim_time
    trj_name = os.path.join(rep_dir, 'trj.dcd')
    log_file = os.path.join(rep_dir, 'sim_log.csv')
    colvar_file = os.path.join(rep_dir, 'COLVAR.npy')

    # Load top and coor in parmed, then in MDAnalysis
    # We can re-use the initial system, because we just want the
    # indices of the protein atoms, and not their coordinates.
    pmd_system = pmd.openmm.load_topology(parm.topology, xyz=coords.positions)
    u = mda.Universe(pmd_system)

    if anchor_residues is None:
        # Get protein anchor points
        prot_com = u.select_atoms('protein and not name H*').center_of_mass()
        x, y, z = prot_com[0], prot_com[1], prot_com[2]

        # Look for protein anchor residues at different radius
        for dist in [5, 10]:
            prot_anchor_atoms = u.select_atoms(f'point {x} {y} {z} {dist} and backbone and not name H*')
            
            if len(prot_anchor_atoms.residues) > 1:
                break
    else:
        sel_str = 'backbone and (%s)' % (' or '.join(['resid %s' % resid for resid in anchor_residues]))
        prot_anchor_atoms = u.select_atoms(sel_str)

    anchor_atom_idx = prot_anchor_atoms.indices.tolist()

    print('Selected residues as protein anchor: %s' % prot_anchor_atoms.residues)

    # Check if we found protein anchors, otherwise we abort.
    # Likely an issue with the periodic conditions
    assert len(anchor_atom_idx) > 0, 'Error: No residues was selected as anchor points.'

    # Get the indices of ligand heavy atoms
    lig = u.select_atoms('resname %s and not element H' % lig_resname)
    lig_ha_idx = lig.indices.tolist()

    # Set up the system to run metadynamics
    system = parm.createSystem(nonbondedMethod=PME, nonbondedCutoff=12 * unit.angstrom, constraints=HBonds, hydrogenMass=4 * unit.amu)

    # Add an 'empty' flat-bottom restraint to fix the issue with PBC.
    # Without one, RMSDForce object fails to account for PBC.
    k = 0 * unit.kilojoules_per_mole  # NOTE - 0 kJ/mol constant
    upper_wall = 10.00 * unit.nanometer
    fb_eq = '(k/2) * max(distance(g1, g2) - upper_wall, 0)^2'
    upper_wall_rest = CustomCentroidBondForce(2, fb_eq)
    upper_wall_rest.addGroup(lig_ha_idx)
    upper_wall_rest.addGroup(anchor_atom_idx)
    upper_wall_rest.addBond([0, 1])
    upper_wall_rest.addGlobalParameter('k', k)
    upper_wall_rest.addGlobalParameter('upper_wall', upper_wall)
    upper_wall_rest.setUsesPeriodicBoundaryConditions(True)
    system.addForce(upper_wall_rest)

    # Set up the typical metadynamics parameters
    alignment_indices = lig_ha_idx + anchor_atom_idx
    rmsd = RMSDForce(coords.positions, alignment_indices)

    grid_min, grid_max = 0.0, 1.0  # nm
    hill_height = set_hill_height * unit.kilocalories_per_mole
    hill_width = 0.002  # nm, also known as sigma

    grid_width = hill_width / 5
    # 'grid' here refers to the number of grid points
    grid = int(abs(grid_min - grid_max) / grid_width)

    rmsd_cv = BiasVariable(rmsd, grid_min, grid_max, hill_width, False, gridWidth=grid)

    # define the metadynamics object
    # deposit bias every 1 ps, BF = 4, write bias every ns
    meta = Metadynamics(system, [rmsd_cv], 300.0 * unit.kelvin, 4.0, hill_height, 250, biasDir=rep_dir, saveFrequency=250000)

    # Set up and run metadynamics
    properties = {"Precision": "mixed"}
    platform = Platform.getPlatformByName('OpenCL')
    integrator = LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.004 * unit.picoseconds)
    simulation = Simulation(parm.topology, system, integrator, platform, properties)
    simulation.context.setPositions(coords.positions)

    simulation.loadState(eq_file_name)

    simulation.reporters.append(DCDReporter(trj_name, 25000))  # every 100 ps
    simulation.reporters.append(StateDataReporter(log_file, 25000,
                                step=True, temperature=True, progress=True, time=True, 
                                potentialEnergy=True, kineticEnergy=True, totalEnergy=True, 
                                remainingTime=True, speed=True, volume=True, density=True,
                                totalSteps=steps, separator=','))  # every 100 ps

    colvar_array = np.array([meta.getCollectiveVariables(simulation)])

    for i in range(0, int(steps), 500):
        if i % 25000 == 0:
            # log the stored COLVAR every 100ps
            np.save(colvar_file, colvar_array)

        meta.step(simulation, 500)

        # record the CVs every 2 ps
        current_cvs = meta.getCollectiveVariables(simulation)
        colvar_array = np.append(colvar_array, [current_cvs], axis=0)
    
    np.save(colvar_file, colvar_array)


def collect_results(nreps, in_dir, out_dir):
    """A function that collects the time-resolved BPM results,
    takes the scores from last 2 ns of the simulation, averages them
    and writes that average as the final score for a given pose.

    Writes a 'results.csv' file in 'out_dir' directory.
    
    Parameters
    ----------
    in_dir : str
        Directory with 'rep_*' directories.
    out_dir : str
        Directory where the 'results.csv' file will be written
    """
    comp = []
    contact = []
    pose = []

    for idx in range(0, nreps):
        f = os.path.join(in_dir, f'rep_{idx}', 'bpm_results.csv')
        df = pd.read_csv(f)
        # Since we only want last 2 ns, get the index of
        # the last 20% of the data points
        last_2ns_idx = round(len(df['comp_score'].values) / 5)  # round up
        comp.append(df['comp_score'].values[-last_2ns_idx:])
        contact.append(df['contact_score'].values[-last_2ns_idx:])
        pose.append(df['pose_score'].values[-last_2ns_idx:])

    # Get the means of the last 2 ns
    mean_comp_score = np.mean(comp)
    mean_pose_score = np.mean(pose)
    mean_contact_score = np.mean(contact)
    # Get the standard deviation of the final 2 ns
    mean_comp_score_std = np.std(comp)
    mean_pose_score_std = np.std(pose)
    mean_contact_score_std = np.std(contact)
    # Format it the Pandas way
    d = {'comp_score': [mean_comp_score], 'comp_score_std': [mean_comp_score_std],
         'pose_score': [mean_pose_score], 'pose_score_std': [mean_pose_score_std],
         'contact_score': [mean_contact_score], 'contact_score_std': [mean_contact_score_std]}

    results_df = pd.DataFrame(data=d)
    results_df = results_df.round(3)
    results_df.to_csv(os.path.join(out_dir, 'results.csv'), index=False)


def cmd_lineparser():
    """ This is executed when run from the command line """
    # Parse the CLI arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=descriptn)

    parser.add_argument("-s", "--structure", type=str, default='solvated.rst7',
                        help='input structure file name (default: %(default)s)')
    parser.add_argument("-p", "--parameters", type=str, default='solvated.prm7',
                        help='input topology file name (default: %(default)s)')
    parser.add_argument("-o", "--output", type=str, default='.',
                        help='output location (default: %(default)s)')
    parser.add_argument("-lig_resname", type=str, default='MOL',
                        help='the name of the ligand (default: %(default)s)')
    parser.add_argument("-anchor_residues", nargs='+', type=int, default=None,
                        help='anchor residues (default: %(default)s)')
    parser.add_argument("-nreps", type=int, default=10,
                        help="number of OpenBPMD repeats (default: %(default)i)")
    parser.add_argument("-hill_height", type=float, default=0.3,
                        help="the hill height in kcal/mol (default: %(default)f)")

    return parser.parse_args()


def main():
    """Main entry point of the app. Takes in argparse.Namespace object as
    a function argument. Carries out a sequence of steps required to obtain a
    stability score for a given ligand pose in the provided structure file.

    1. Load the structure and parameter files.
    2. If absent, create an output folder.
    3. Minimization up to ener. tolerance of 10 kJ/mol.
    4. 500 ps equilibration in NVT ensemble with position
       restraints on solute heavy atoms with the force 
       constant of 5 kcal/mol/A^2
    5. Run NREPs (default=10) of binding pose metadynamics simulations,
       writing trajectory files and a time-resolved BPM scores for each
       repeat.
    6. Collect results from the OpenBPMD simulations and
       write a final score for a given protein-ligand
       structure.

    Parameters
    ----------
    args.structure : str, default='solvated.rst7'
        Name of the structure file, either Amber or Gromacs format.
    args.parameters : str, default='solvated.prm7'
        Name of the parameter or topology file, either Amber or Gromacs
        format.
    args.output : str, default='.'
        Path to and the name of the output directory.
    args.lig_resname : str, default='LIG'
        Residue name of the ligand in the structure/parameter file.
    args.nreps : int, default=10
        Number of repeat OpenBPMD simulations to run in series.
    args.hill_height : float, default=0.3
        Size of the metadynamical hill, in kcal/mol.
    """
    args = cmd_lineparser()

    if args.structure.endswith('.gro'):
        coords = GromacsGroFile(args.structure)
        box_vectors = coords.getPeriodicBoxVectors()
        parm = GromacsTopFile(args.parameters, periodicBoxVectors=box_vectors)
    else:
        coords = AmberInpcrdFile(args.structure)
        parm = AmberPrmtopFile(args.parameters)

    if not os.path.isdir(f'{args.output}'):
        os.mkdir(f'{args.output}')

    # Minimize
    min_file_name = os.path.join(args.output, 'min.xml')
    if not os.path.isfile(min_file_name):
        minimisation(parm, coords, min_file_name)

    # ..and Equilibrate
    eq_file_name = os.path.join(args.output, 'equil.xml')
    if not os.path.isfile(eq_file_name):
        equilibrate(parm, coords, min_file_name, eq_file_name, args.lig_resname)

    # Run NREPS number of production simulations
    for idx in range(0, args.nreps):
        print("Producing... (run: %02d/%02d)" % (idx + 1, args.nreps))

        rep_dir = os.path.join(args.output, f'rep_{idx}')
        if not os.path.isdir(rep_dir):
            os.mkdir(rep_dir)

        trj_name = os.path.join(rep_dir, 'trj.dcd')

        if os.path.isfile(os.path.join(rep_dir, 'bpm_results.csv')):
            continue
        
        produce(parm, coords, args.lig_resname, args.anchor_residues, args.hill_height, rep_dir, eq_file_name)

        # center everything using MDTraj, to fix any PBC imaging issues
        # mdtraj can't use GMX TOP, so we have to specify the GRO file instead
        if args.structure.endswith('.gro'):
            mdtraj_top = args.structure
        else:
            mdtraj_top = args.parameters

        mdu = md.load(trj_name, top=mdtraj_top)
        mdu.image_molecules()
        mdu.save(trj_name)
                
        pose_scores = get_pose_score(args.parameters, trj_name, args.lig_resname)
        contact_scores = get_contact_score(args.parameters, trj_name, args.lig_resname)

        # Calculate the CompScore at every frame
        comp_scores = pose_scores - 5 * contact_scores
        scores = np.stack((comp_scores, pose_scores, contact_scores), axis=-1)

        # Save a DataFrame to CSV
        df = pd.DataFrame(data=scores, columns=['comp_score', 'pose_score', 'contact_score'])
        df.to_csv(os.path.join(rep_dir, 'bpm_results.csv'), index=False)
                
    collect_results(args.nreps, args.output, args.output)


if __name__ == '__main__':
    main()
