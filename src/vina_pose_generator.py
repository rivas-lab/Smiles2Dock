import os
import tempfile
from typing import Tuple, Optional, Union, List
import platform
import logging
import numpy as np
from subprocess import Popen, PIPE
from typing import List, Optional, Tuple, Union
import time

from deepchem.dock.binding_pocket import BindingPocketFinder
from deepchem.utils.data_utils import download_url, get_data_dir
from deepchem.utils.typing import RDKitMol
from deepchem.utils.geometry_utils import compute_centroid, compute_protein_range
from deepchem.utils.rdkit_utils import load_molecule, write_molecule
from deepchem.utils.docking_utils import load_docked_ligands, write_vina_conf, write_gnina_conf, read_gnina_log

logging.getLogger("deepchem").setLevel(logging.ERROR)  
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s\n', level=logging.INFO)

logger = logging.getLogger(__name__)
DOCKED_POSES = List[Tuple[RDKitMol, RDKitMol]]

class PoseGenerator(object):
    """A Pose Generator computes low energy conformations for molecular complexes.

    Many questions in structural biophysics reduce to that of computing
    the binding free energy of molecular complexes. A key step towards
    computing the binding free energy of two complexes is to find low
    energy "poses", that is energetically favorable conformations of
    molecules with respect to each other. One application of this
    technique is to find low energy poses for protein-ligand
    interactions.
    """

    def generate_poses(self,
                       molecular_complex: Tuple[str, str],
                       centroid: Optional[np.ndarray] = None,
                       box_dims: Optional[np.ndarray] = None,
                       exhaustiveness: int = 10,
                       num_modes: int = 9,
                       num_pockets: Optional[int] = None,
                       out_dir: Optional[str] = None,
                       generate_scores: bool = False):
        """Generates a list of low energy poses for molecular complex

        Parameters
        ----------
        molecular_complexes: Tuple[str, str]
            A representation of a molecular complex. This tuple is
            (protein_file, ligand_file).
        centroid: np.ndarray, optional (default None)
            The centroid to dock against. Is computed if not specified.
        box_dims: np.ndarray, optional (default None)
            A numpy array of shape `(3,)` holding the size of the box to dock.
            If not specified is set to size of molecular complex plus 5 angstroms.
        exhaustiveness: int, optional (default 10)
            Tells pose generator how exhaustive it should be with pose
            generation.
        num_modes: int, optional (default 9)
            Tells pose generator how many binding modes it should generate at
            each invocation.
        num_pockets: int, optional (default None)
            If specified, `self.pocket_finder` must be set. Will only
            generate poses for the first `num_pockets` returned by
            `self.pocket_finder`.
        out_dir: str, optional (default None)
            If specified, write generated poses to this directory.
        generate_score: bool, optional (default False)
            If `True`, the pose generator will return scores for complexes.
            This is used typically when invoking external docking programs
            that compute scores.

        Returns
        -------
        A list of molecular complexes in energetically favorable poses.
        """
        raise NotImplementedError
        
class VinaPoseGenerator(PoseGenerator):
    """Uses Autodock Vina to generate binding poses.

    This class uses Autodock Vina to make make predictions of
    binding poses.

    Example
    -------
    >> import deepchem as dc
    >> vpg = dc.dock.VinaPoseGenerator(pocket_finder=None)
    >> protein_file = '1jld_protein.pdb'
    >> ligand_file = '1jld_ligand.sdf'
    >> poses, scores = vpg.generate_poses(
    ..        (protein_file, ligand_file),
    ..        exhaustiveness=1,
    ..        num_modes=1,
    ..        out_dir=tmp,
    ..        generate_scores=True)

    Note
    ----
    This class requires RDKit and vina to be installed. As on 9-March-22,
    Vina is not available on Windows. Hence, this utility is currently
    available only on Ubuntu and MacOS.
    """

    def __init__(self, pocket_finder: Optional[BindingPocketFinder] = None):
        """Initializes Vina Pose Generator

        Parameters
        ----------
        pocket_finder: BindingPocketFinder, optional (default None)
            If specified should be an instance of
            `dc.dock.BindingPocketFinder`.
        """
        self.pocket_finder = pocket_finder

    def generate_poses(
            self,
            molecular_complex: Tuple[str, str],
            exhaustiveness: int = 10,
            num_modes: int = 9,
            num_pockets: Optional[int] = None,
            out_dir: Optional[str] = None,
            generate_scores: Optional[bool] = False,
            ligand_name = None,
            **kwargs) -> Union[Tuple[DOCKED_POSES, List[float]], DOCKED_POSES]:
        """Generates the docked complex and outputs files for docked complex.

        Parameters
        ----------
        molecular_complexes: Tuple[str, str]
            A representation of a molecular complex. This tuple is
            (protein_file, ligand_file). The protein should be a pdb file
            and the ligand should be an sdf file.
        exhaustiveness: int, optional (default 10)
            Tells Autodock Vina how exhaustive it should be with pose generation. A
            higher value of exhaustiveness implies more computation effort for the
            docking experiment.
        num_modes: int, optional (default 9)
            Tells Autodock Vina how many binding modes it should generate at
            each invocation.
        num_pockets: int, optional (default None)
            If specified, `self.pocket_finder` must be set. Will only
            generate poses for the first `num_pockets` returned by
            `self.pocket_finder`.
        out_dir: str, optional
            If specified, write generated poses to this directory.
        generate_score: bool, optional (default False)
            If `True`, the pose generator will return scores for complexes.
            This is used typically when invoking external docking programs
            that compute scores.
        kwargs:
            The kwargs - cpu, min_rmsd, max_evals, energy_range supported by VINA
            are as documented in https://autodock-vina.readthedocs.io/en/latest/vina.html

        Returns
        -------
        Tuple[`docked_poses`, `scores`] or `docked_poses`
            Tuple of `(docked_poses, scores)` or `docked_poses`. `docked_poses`
            is a list of docked molecular complexes. Each entry in this list
            contains a `(protein_mol, ligand_mol)` pair of RDKit molecules.
            `scores` is a list of binding free energies predicted by Vina.

        Raises
        ------
        `ValueError` if `num_pockets` is set but `self.pocket_finder is None`.
        """
        if "cpu" in kwargs:
            cpu = kwargs["cpu"]
        else:
            cpu = 0
        if "min_rmsd" in kwargs:
            min_rmsd = kwargs["min_rmsd"]
        else:
            min_rmsd = 1.0
        if "max_evals" in kwargs:
            max_evals = kwargs["max_evals"]
        else:
            max_evals = 0
        if "energy_range" in kwargs:
            energy_range = kwargs["energy_range"]
        else:
            energy_range = 3.0

        try:
            from vina import Vina
        except ModuleNotFoundError:
            raise ImportError("This function requires vina to be installed")

        if out_dir is None:
            out_dir = tempfile.mkdtemp()

        if num_pockets is not None and self.pocket_finder is None:
            raise ValueError(
                "If num_pockets is specified, pocket_finder must have been provided at construction time."
            )

        print(self.pocket_finder)
        # Parse complex
        if len(molecular_complex) > 2:
            raise ValueError(
                "Autodock Vina can only dock protein-ligand complexes and not more general molecular complexes."
            )

        protein_file, ligand_file = molecular_complex
        
        assert self.pocket_finder is not None, 'Pocket finder is None.'

        logger.info("About to find putative binding pockets")
        pockets = self.pocket_finder.find_pockets(protein_file)
        logger.info("%d pockets found in total" % len(pockets))
        logger.info("Computing centroid and size of proposed pockets.")
        centroids, dimensions = [], []
        for pocket in pockets:
            (x_min, x_max), (y_min, y_max), (
                z_min,
                z_max) = pocket.x_range, pocket.y_range, pocket.z_range

            x_box = (x_max - x_min) / 2.
            y_box = (y_max - y_min) / 2.
            z_box = (z_max - z_min) / 2.
            centroids.append(pocket.center())
            dimensions.append(np.array((x_box, y_box, z_box)))

        if num_pockets is not None:
            logger.info(
                "num_pockets = %d so selecting this many pockets for docking." %
                num_pockets)
            centroids = centroids[:num_pockets]
            dimensions = dimensions[:num_pockets]
                    
        docked_complexes = []
        all_scores = []
        vpg = Vina(sf_name='vina',
                   cpu=cpu,
                   seed=0,
                   no_refine=False,
                   verbosity=1)
        for i, (protein_centroid,
                box_dims) in enumerate(zip(centroids, dimensions)):
            logger.info("Docking in pocket %d/%d" % (i + 1, len(centroids)))
            logger.info("Docking with center: %s" % str(protein_centroid))
            logger.info("Box dimensions: %s" % str(box_dims))

            # Define locations of output files
            logger.info("About to call Vina")

            vpg.set_receptor(protein_file)
            logger.info('Protein file')
            logger.info(protein_file)
            vpg.set_ligand_from_file(ligand_file)
            
            vpg.compute_vina_maps(center=protein_centroid, box_size=box_dims)
            
            vpg.dock(exhaustiveness=exhaustiveness,
                     n_poses=num_modes,
                     min_rmsd=min_rmsd,
                     max_evals=max_evals)
            end_p_prep = time.time()
            duration = end_p_prep - start_p_prep
            logger.info(f"Docking with Deepchem took {duration} seconds.")
            
            energies = vina.energies(n_poses=n_poses, energy_range=energy_range)
            print(energies)
            all_scores += energies

        if generate_scores:
            return docked_complexes, all_scores
        else:
            return docked_complexes
        