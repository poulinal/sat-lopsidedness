# ADP 2026

from myproject.utilities.Subhalo import Subhalo
from .GalaxyGroup import GalaxyGroup
from .parallelTools import parallel_map, get_optimal_processes
import h5py as h5
import numpy as np
from typing import Optional

class ListGalaxyGroup:
    """
    Class to manage a list of GalaxyGroup objects and perform analyses on them.
    
    Can be initialized from an HDF5 file using the from_hdf5 class method or by providing a list of GalaxyGroup objects and header information.
    
    Attributes
    ----------
    listGalaxyGroups : list[GalaxyGroup]
        A list containing GalaxyGroup objects.
    headerInformation : dict
        A dictionary to store header information about the dataset.
    list_pairwise_differences : list[list[float]]
        A list to store pairwise polar angle differences between satellite galaxies in each galaxy group.
    MRL_values : list[float]
        A list to store Mean Resultant Length (MRL) directionality values for each galaxy group.
    Methods
    -------
    addGalaxyGroup(galaxyGroup : GalaxyGroup)
        Adds a GalaxyGroup object to the list.
    getNumGalaxyGroups() -> int
        Returns the number of GalaxyGroup objects in the list.
    getListPairwiseDifferences() -> list[list[float]]
        Returns the list of pairwise polar angle differences.
    compute_all_pairwise_polar_differences() -> list[list[float]]
        Computes all pairwise polar angle differences between satellite galaxies in each galaxy group.
    compute_probablity_distribution_of_polar_differences(bin_size : float=5.0) -> tuple[np.ndarray, np.ndarray]
        Computes the probability distribution of polar angle differences.
    compute_all_MRL_directionality() -> list[float]
        Computes the Mean Resultant Length (MRL) directionality for each galaxy group.
    compute_probablity_distribution_of_MRL_directionality(bin_size : float=0.05) -> tuple[np.ndarray, np.ndarray]
        Computes the probability distribution of MRL directionality.
    filterSubhalos(minStellarMass : float=None, maxStellarMass : float=None, minHalfMassRad_kpc : float=None, maxHalfMassRad_kpc : float=None, centralPosTolerance_kpc : float=1000) -> None
        Filters subhalos in each galaxy group based on specified criteria.
    correctPositions(boxsize : float) -> None
        Corrects the positions of subhalos in each galaxy group to account for periodic boundary conditions.
    save_to_hdf5(h5file : h5.File, overwrite : bool=True) -> None
        Saves the ListGalaxyGroup data to an HDF5 file.
    load_from_hdf5(h5file : h5.File) -> None
        Loads the ListGalaxyGroup data from an HDF5 file.
    """
    def __init__(self, listGalaxyGroups : list[GalaxyGroup]=[], headerInformation : dict={}):
        self.listGalaxyGroups = listGalaxyGroups
        self.headerInformation = headerInformation
        self.list_pairwise_differences : list[list[float]] = []
        self.MRL_values : list[float] = []

        self.lenGalaxyGroups = len(self.listGalaxyGroups)
        
    @classmethod
    def from_hdf5(cls, h5file : h5.File):
        instance = cls()
        instance.load_from_hdf5(h5file)
        return instance
        
    def addGalaxyGroup(self, galaxyGroup : GalaxyGroup):
        self.listGalaxyGroups.append(galaxyGroup)
        self.lenGalaxyGroups += 1
        
    def setGalaxyGroups(self, listGalaxyGroups : list[GalaxyGroup]):
        self.listGalaxyGroups = listGalaxyGroups
        self.lenGalaxyGroups = len(self.listGalaxyGroups)
        
    def getNumGalaxyGroups(self):
        return self.lenGalaxyGroups
    
    def getAllGalaxyGroups(self):
        return self.listGalaxyGroups
    
    def getGalaxyGroupI(self, i):
        return self.listGalaxyGroups[i]
    
    def getListPairwiseDifferences(self) -> list[list[tuple[float, float, float]]]:
        return self.list_pairwise_differences
            
    def compute_all_pairwise_polar_differences(self, parallelize : bool=False, n_processes : Optional[int]=None) -> list[list[tuple[float, float, float]]]:
        '''
        Docstring for compute_all_pairwise_polar_differences
        Computes all pairwise polar angle differences (in each plane: XY, YZ, ZX) between satellite galaxies in each galaxy group with resepect to the host galaxy.
        Returns a list of all pairwise polar angle differences. Indicies go from 0 to 180º. Indicie of 0 means aligned, 180º means anti-aligned. Values correspond to the density/probability of finding satellite galaxies at a given polar angle difference.
        
        Note: Each entry in the returned list corresponds to a galaxy group, containing a list of tuples. Each tuple contains the polar angle differences (in degrees) between a pair of satellite galaxies in the XY, YZ, and ZX planes respectively. NOT the average of the three planes.
        
        :param self: Description
        :return: Returns a list where each indicie corresponds to a galaxy group, each containing a list of pairwise polar angle differences between satellite galaxies.
        :rtype: list[list[float]]
        '''
        self.list_pairwise_differences = []
        if parallelize:
            if n_processes is None:
                n_processes = get_optimal_processes(len(self.listGalaxyGroups))
        
            print(f"Computing pairwise differences in parallel with {n_processes} processes...")
            self.list_pairwise_differences = parallel_map(
                ListGalaxyGroup._compute_pairwise_for_group,
                self.listGalaxyGroups,
                n_processes=n_processes,
                show_progress=True
            )
        else:
            for galaxyGroup in self.listGalaxyGroups:
                print(f"Progress: Processing Galaxy Group ID {galaxyGroup.getGroupID()} / {len(self.listGalaxyGroups)}", end='\r')
                group_pairwise_differences = []
                group_pairwise_differences = ListGalaxyGroup._compute_pairwise_for_group(galaxyGroup)

                self.list_pairwise_differences.append(group_pairwise_differences)
            
        return self.list_pairwise_differences
    
    def compute_probablity_distribution_of_polar_differences(self, bin_size : float=5.0, parallelize: bool = False) -> tuple[np.ndarray, np.ndarray]:
        '''
        Docstring for compute_probablity_distribution_of_polar_differences
        Computes the probability distribution of polar angle differences between satellite galaxies in each galaxy group with resepect to the host galaxy.
        Returns a tuple containing the bin centers and the corresponding probability densities.
        
        :param self: Description
        :param bin_size: Size of the bins for the histogram (default is 5.0 degrees)
        :return: Tuple of (bin_centers, probability_densities)
        :rtype: tuple[np.ndarray, np.ndarray]
        '''
        if not self.list_pairwise_differences:
            self.compute_all_pairwise_polar_differences(parallelize=parallelize, n_processes=None)
        pairwise_differences_flatten = []
        for galaxyPairwiseGroup in self.list_pairwise_differences:
            for pair in galaxyPairwiseGroup:
                pairwise_differences_flatten.extend(pair)  # Unpack the tuple and add each angle difference
        bins = np.arange(0, 180 + bin_size, bin_size)
        hist, bin_edges = np.histogram(pairwise_differences_flatten, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, hist
    
    def compute_all_MRL_directionality(self, parallelize: bool = False, n_processes: Optional[int] = None) -> list[float]:
        '''
        Docstring for compute_MRL_directionality
        Computes the Mean Resultant Length (MRL) directionality for each galaxy group.
        Returns a list of MRL directionality values for each galaxy group. Indicie goes from 0 to 1, where 0 indicates isotropic distribution and 1 indicates satellites clustering toward one direction. Values correspond to the density/probability of finding satellite galaxies aligned in a in the MRL.
        
        R = (1/N) * sqrt( (sum(cos(theta_i)))^2 + (sum(sin(theta_i)))^2 )
        
        Note: This implementation computes MRL for each of the three planes (XY, YZ, ZX) separately and appends all three values to the MRL_values list, NOT their average.
        
        :param self: Description
        :return: Returns a list of MRL directionality values for each galaxy group.
        :rtype: list[float]
        '''
        self.MRL_values = []
        if not self.list_pairwise_differences:
            self.compute_all_pairwise_polar_differences(parallelize=parallelize, n_processes=n_processes)
        else:
            print("Using pre-computed pairwise polar differences.")
            
        if parallelize:
            print(f"Computing MRL directionality in parallel with {n_processes} processes...")
            results = parallel_map(
                ListGalaxyGroup._compute_MRL_for_group,
                self.list_pairwise_differences,
                n_processes=n_processes,
                show_progress=True
            )
            
            # Flatten results (each result is [R_xy, R_yz, R_zx])
            self.MRL_values = []
            for mrl_vals in results:
                self.MRL_values.extend(mrl_vals)
        else:
            for galaxyPairwiseGroup in self.getListPairwiseDifferences():
                n = len(galaxyPairwiseGroup)
                list_diff_xy = []
                list_diff_yz = []
                list_diff_zx = []
                
                for pairwise_difference in galaxyPairwiseGroup:
                    diff_xy, diff_yz, diff_zx = pairwise_difference
                    list_diff_xy.append(diff_xy)
                    list_diff_yz.append(diff_yz)
                    list_diff_zx.append(diff_zx)
                    
                # Compute MRL for XY plane
                cosComponent = np.sum([np.cos(np.radians(angle)) for angle in list_diff_xy])
                sinComponent = np.sum([np.sin(np.radians(angle)) for angle in list_diff_xy])
                R_xy = (1/n) * np.sqrt(cosComponent**2 + sinComponent**2) if n > 0 else 0.0
                # Compute MRL for YZ plane
                cosComponent = np.sum([np.cos(np.radians(angle)) for angle in list_diff_yz])
                sinComponent = np.sum([np.sin(np.radians(angle)) for angle in list_diff_yz])
                R_yz = (1/n) * np.sqrt(cosComponent**2 + sinComponent**2) if n > 0 else 0.0
                # Compute MRL for ZX plane
                cosComponent = np.sum([np.cos(np.radians(angle)) for angle in list_diff_zx])
                sinComponent = np.sum([np.sin(np.radians(angle)) for angle in list_diff_zx])
                R_zx = (1/n) * np.sqrt(cosComponent**2 + sinComponent**2) if n > 0 else 0.0
                # Don't take average, append all three values
                self.MRL_values.extend([R_xy, R_yz, R_zx])
        return self.MRL_values
            
    def compute_probablity_distribution_of_MRL_directionality(self, bin_size : float=0.05) -> tuple[np.ndarray, np.ndarray]:
        '''
        Docstring for compute_probablity_distribution_of_MRL_directionality
        Computes the probability distribution of Mean Resultant Length (MRL) directionality for each galaxy group.
        Returns a tuple containing the bin centers and the corresponding probability densities.
        
        :param self: Description
        :param bin_size: Size of the bins for the histogram (default is 0.05)
        :return: Tuple of (bin_centers, probability_densities)
        :rtype: tuple[np.ndarray, np.ndarray]
        '''
        if not self.MRL_values:
            self.compute_all_MRL_directionality()
        else:
            print("Using pre-computed MRL directionality values.")
        bins = np.arange(0, 1 + bin_size, bin_size)
        hist, bin_edges = np.histogram(self.MRL_values, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, hist
        
    def filterSubhalos(self, minGGMass : float=None, maxGGMass : float=None, minSatStellarMass : float=None, maxSatStellarMass : float=None, minHalfMassRad_kpc : float=None, maxHalfMassRad_kpc : float=None, centralPosTolerance_kpc : float=1000, parallelize : bool=False, n_processes: Optional[int]=None) -> None:
        '''
        Modifies list_galaxy_groups and Filters subhalos in each galaxy group based on specified criteria.
        - remove non cosmlogoical in origin (subhaloflag = 0)
        - only mass greater than 13 solar mass
        - remove groups where no single central (i.e. central pos does not match cm pos)
        - Later on will need to filter based on radius due to smoothing length: because the smoothing length used by the TNG300 is different for z < 1 and z >= 1 we'll likely want to adopt a minimum  size for the galaxies we want to use following Curtis et al. (2026) (note: we need to check to see if TNGCluster used the same smoothing length change as TNG300)
        
        :param minSatStellarMass: Minimum stellar mass to retain a subhalo (default is None)
        :param maxSatStellarMass: Maximum stellar mass to retain a subhalo (default is None)
        :param minHalfMassRad_kpc: Minimum half-mass radius in kpc to retain a subhalo (default is None)
        :param maxHalfMassRad_kpc: Maximum half-mass radius in kpc to retain a subhalo (default is None)
        :param centralPosTolerance_kpc: Maximum distance from central position in kpc to retain a subhalo (default is 1000 kpc, i.e. 1 Mpc)
        :param parallelize: Whether to parallelize the filtering process (default is False)
        :param n_processes: Number of processes to use if parallelizing (default is None, which uses optimal number)
        '''
        list_filtered_galaxy_groups : list[GalaxyGroup]= []
        total = len(self.listGalaxyGroups)
            
        # Prepare arguments for parallel processing
        args_list = [
            (gg, minGGMass, maxGGMass, minSatStellarMass, maxSatStellarMass, minHalfMassRad_kpc, maxHalfMassRad_kpc, centralPosTolerance_kpc)
            for gg in self.listGalaxyGroups
        ]
        
        if parallelize:
            if n_processes is None:
                n_processes = get_optimal_processes(len(self.listGalaxyGroups))
            
            print(f"Filtering subhalos in parallel with {n_processes} processes...")
            
            # Use imap to get results as they complete (allows progress tracking)
            # imap doesn't unpack tuples, unlike istarmap
            import multiprocessing as mp
            with mp.Pool(processes=n_processes) as pool:
                results = []
                for i, result in enumerate(pool.imap(ListGalaxyGroup._filter_subhalos_for_group, args_list), 1):
                    results.append(result)
                    percent = (i / total) * 100
                    print(f"\rProgress: {i}/{total} ({percent:.1f}%)", end='', flush=True)
                print()  # New line after progress
            
            # Filter out None results (skipped groups)
            self.setGalaxyGroups([gg for gg in results if gg is not None])
            
            print(f"After filtering: {self.lenGalaxyGroups} galaxy groups retained.")
        else:
            list_filtered_galaxy_groups : list[GalaxyGroup]= []
            for args in args_list:
                print(f"Progress: Processing Galaxy Group ID {args[0].getGroupID()} / {len(self.listGalaxyGroups)}", end='\r')
                list_filtered_galaxy_groups.append(ListGalaxyGroup._filter_subhalos_for_group(args))
                    
            self.setGalaxyGroups([gg for gg in list_filtered_galaxy_groups if gg is not None])
            
        return self.getAllGalaxyGroups()
                        
    def correctPositions(self, boxsize : float, parallelize : bool=False, n_processes: Optional[int]=None) -> None:
        '''
        Corrects the positions of subhalos in each galaxy group to account for periodic boundary conditions. 
        Ensure all positions are relative to the central galaxy.
        
        :param boxsize: Size of the simulation box
        '''
        total = len(self.listGalaxyGroups)
            
        # Prepare arguments for parallel processing
        args_list = [(gg, boxsize) for gg in self.listGalaxyGroups]
        
        if parallelize:
            if n_processes is None:
                n_processes = get_optimal_processes(len(self.listGalaxyGroups))
            
            print(f"Correcting positions in parallel with {n_processes} processes...")
            
            # Use imap to get results as they complete (allows progress tracking)
            import multiprocessing as mp
            with mp.Pool(processes=n_processes) as pool:
                results = []
                for i, result in enumerate(pool.imap(ListGalaxyGroup._correct_positions_for_group, args_list), 1):
                    results.append(result)
                    percent = (i / total) * 100
                    print(f"\rProgress: {i}/{total} ({percent:.1f}%)", end='', flush=True)
                print()  # New line after progress
            
            self.setGalaxyGroups(results)
        else:
            results = []
            for args in args_list:
                print(f"Progress: Processing Galaxy Group ID {args[0].getGroupID()} / {len(self.listGalaxyGroups)}", end='\r')
                results.append(ListGalaxyGroup._correct_positions_for_group(args))
                
            self.setGalaxyGroups(results)
        
        return self.getAllGalaxyGroups()
       
       
       
    def save_to_hdf5(self, h5file: h5.File, overwrite: bool = True, parallize : bool = False, n_processes: Optional[int] = None):
        '''
        Parallel version of save_to_hdf5 using multiprocessing.
        Serializes galaxy group data in parallel, then writes sequentially.
        Most useful when you have many galaxy groups with complex data.
        
        Note: The actual HDF5 writing is still sequential (HDF5 limitation),
        but data preparation is parallelized.
        
        Parameters
        ----------
        h5file : h5.File
            HDF5 file handle to write to
        overwrite : bool
            If True, create new groups; if False, skip existing groups
        n_processes : int, optional
            Number of processes to use (default: CPU count - 1)
        '''
        if not overwrite:
            # Serial processing for append mode
            existing_groups = set(h5file['GalaxyGroups'].keys()) if 'GalaxyGroups' in h5file else set()
            for i, galaxyGroup in enumerate(self.listGalaxyGroups):
                if f'GalaxyGroup_{i}' in existing_groups:
                    print(f"GalaxyGroup_{i} already exists in file. Skipping.")
                    continue
        else:
            # Write header information
            for key, value in self.headerInformation.items():
                h5file.attrs[key] = value
            
            total = len(self.listGalaxyGroups)
            
            if parallize and n_processes is None:
                n_processes = get_optimal_processes(len(self.listGalaxyGroups))
            
                print(f"Serializing galaxy group data in parallel with {n_processes} processes...")
            
            # Prepare arguments for parallel processing
            args_list = [(gg, i) for i, gg in enumerate(self.listGalaxyGroups)]
            
            if parallize:
                # Use imap to get results as they complete (allows progress tracking)
                import multiprocessing as mp
                with mp.Pool(processes=n_processes) as pool:
                    serialized_data = []
                    for i, result in enumerate(pool.imap(ListGalaxyGroup._serialize_galaxy_group, args_list), 1):
                        serialized_data.append(result)
                        percent = (i / total) * 100
                        print(f"\rSerialization: {i}/{total} ({percent:.1f}%)", end='', flush=True)
                    print()  # New line after progress
            else:
                print("Serializing galaxy group data sequentially...")
                serialized_data = []
                for i, args in enumerate(args_list, 1):
                    result = ListGalaxyGroup._serialize_galaxy_group(args)
                    serialized_data.append(result)
                    percent = (i / total) * 100
                    print(f"\rSerialization: {i}/{total} ({percent:.1f}%)", end='', flush=True)
                print()  # New line after progress
            
            print("Writing serialized data to HDF5 file...")
            grp = h5file.create_group('GalaxyGroups')
            total = len(serialized_data)
            
            for idx, group_data in enumerate(serialized_data, 1):
                i = group_data['index']
                percent = (idx / total) * 100
                print(f"\rWriting: {idx}/{total} ({percent:.1f}%)", end='', flush=True)
                
                gg_grp = grp.create_group(f'GalaxyGroup_{i}')
                gg_grp.attrs['group_id'] = group_data['group_id']
                gg_grp.attrs['MCrit200'] = group_data['MCrit200']
                gg_grp.attrs['posCM'] = group_data['posCM']
                gg_grp.attrs['pos'] = group_data['pos']
                
                subhalos_grp = gg_grp.create_group('Subhalos')
                for j, subhalo_data in enumerate(group_data['subhalos']):
                    sh_grp = subhalos_grp.create_group(f'Subhalo_{j}')
                    sh_grp.attrs['idx'] = subhalo_data['idx']
                    sh_grp.attrs['group_id'] = subhalo_data['group_id']
                    sh_grp.attrs['flag'] = subhalo_data['flag']
                    sh_grp.attrs['mass'] = subhalo_data['mass']
                    sh_grp.attrs['stellarMass'] = subhalo_data['stellarMass']
                    sh_grp.attrs['groupNumber'] = subhalo_data['groupNumber']
                    sh_grp.attrs['position'] = subhalo_data['position']
                    sh_grp.attrs['halfMassRad'] = subhalo_data['halfMassRad']
                    sh_grp.attrs['vmaxRadius'] = subhalo_data['vmaxRadius']
                    sh_grp.attrs['luminosities'] = subhalo_data['luminosities']
            
            print(f"\nCompleted writing {len(serialized_data)} galaxy groups to HDF5.")
                    
    def load_from_hdf5(self, h5file : h5.File):
        self.listGalaxyGroups = []
        self.headerInformation = {}
        for key, value in h5file.attrs.items():
            self.headerInformation[key] = value
        grp = h5file['GalaxyGroups']
        for i, gg_key in enumerate(grp):
            print(f"Progress: {i+1}/{len(grp)}", end='\r')
            gg_grp : h5.Group = grp[gg_key]
            galaxy_group_id = gg_grp.attrs['group_id']
            MCrit200 = gg_grp.attrs['MCrit200']
            posCM = gg_grp.attrs['posCM']
            pos = gg_grp.attrs['pos']
            galaxyGroup = GalaxyGroup(galaxy_group_id, MCrit200, posCM, pos)
            
            subhalos_grp : h5.Group = gg_grp['Subhalos']
            for sh_key in subhalos_grp:
                sh_grp : h5.Group = subhalos_grp[sh_key]
                idx = sh_grp.attrs['idx']
                group_id = sh_grp.attrs['group_id']
                flag = sh_grp.attrs['flag']
                mass = sh_grp.attrs['mass']
                stellarMass = sh_grp.attrs['stellarMass']
                groupNumber = sh_grp.attrs['groupNumber']
                position = sh_grp.attrs['position']
                halfMassRad = sh_grp.attrs['halfMassRad']
                vmaxRadius = sh_grp.attrs['vmaxRadius']
                luminosities = sh_grp.attrs['luminosities']
                
                subhalo = Subhalo(idx, group_id, flag, mass, stellarMass, groupNumber, position, halfMassRad, vmaxRadius, luminosities)
                galaxyGroup.addSubhalo(subhalo)
            
            self.addGalaxyGroup(galaxyGroup)
        print(f"\nLoaded {len(self.listGalaxyGroups)} galaxy groups from HDF5.")


    # Standalone functions for multiprocessing (must be picklable)
    @staticmethod
    def _compute_pairwise_for_group(galaxyGroup):
        """Helper function to compute pairwise differences for a single galaxy group."""
        from myproject.utilities.Subhalo import Subhalo
        import numpy as np
        
        subhalos = galaxyGroup.getSubhalos()
        num_subhalos = len(subhalos)
        central_pos = galaxyGroup.getPos()
        group_pairwise_differences = []
        
        for i in range(num_subhalos):
            for j in range(i + 1, num_subhalos):
                pos_i = subhalos[i].getPosition()
                pos_j = subhalos[j].getPosition()
                
                # XY plane
                vec_i_xy = np.array([pos_i[0] - central_pos[0], pos_i[1] - central_pos[1]])
                vec_j_xy = np.array([pos_j[0] - central_pos[0], pos_j[1] - central_pos[1]])
                angle_i_xy = np.arctan2(vec_i_xy[1], vec_i_xy[0])
                angle_j_xy = np.arctan2(vec_j_xy[1], vec_j_xy[0])
                diff_xy = np.abs(angle_i_xy - angle_j_xy) * (180.0 / np.pi)
                diff_xy = diff_xy if diff_xy <= 180 else 360 - diff_xy
                
                # YZ plane
                vec_i_yz = np.array([pos_i[1] - central_pos[1], pos_i[2] - central_pos[2]])
                vec_j_yz = np.array([pos_j[1] - central_pos[1], pos_j[2] - central_pos[2]])
                angle_i_yz = np.arctan2(vec_i_yz[1], vec_i_yz[0])
                angle_j_yz = np.arctan2(vec_j_yz[1], vec_j_yz[0])
                diff_yz = np.abs(angle_i_yz - angle_j_yz) * (180.0 / np.pi)
                diff_yz = diff_yz if diff_yz <= 180 else 360 - diff_yz
                
                # ZX plane
                vec_i_zx = np.array([pos_i[2] - central_pos[2], pos_i[0] - central_pos[0]])
                vec_j_zx = np.array([pos_j[2] - central_pos[2], pos_j[0] - central_pos[0]])
                angle_i_zx = np.arctan2(vec_i_zx[1], vec_i_zx[0])
                angle_j_zx = np.arctan2(vec_j_zx[1], vec_j_zx[0])
                diff_zx = np.abs(angle_i_zx - angle_j_zx) * (180.0 / np.pi)
                diff_zx = diff_zx if diff_zx <= 180 else 360 - diff_zx
                
                pairwise_difference = (diff_xy, diff_yz, diff_zx)
                group_pairwise_differences.append(pairwise_difference)
        
        return group_pairwise_differences

    @staticmethod
    def _compute_MRL_for_group(pairwise_group):
        """Helper function to compute MRL for a single galaxy group's pairwise differences."""
        import numpy as np
        
        n = len(pairwise_group)
        if n == 0:
            return [0.0, 0.0, 0.0]
        
        list_diff_xy = []
        list_diff_yz = []
        list_diff_zx = []
        
        for pairwise_difference in pairwise_group:
            diff_xy, diff_yz, diff_zx = pairwise_difference
            list_diff_xy.append(diff_xy)
            list_diff_yz.append(diff_yz)
            list_diff_zx.append(diff_zx)
        
        # Compute MRL for XY plane
        cosComponent = np.sum([np.cos(np.radians(angle)) for angle in list_diff_xy])
        sinComponent = np.sum([np.sin(np.radians(angle)) for angle in list_diff_xy])
        R_xy = (1/n) * np.sqrt(cosComponent**2 + sinComponent**2)
        
        # Compute MRL for YZ plane
        cosComponent = np.sum([np.cos(np.radians(angle)) for angle in list_diff_yz])
        sinComponent = np.sum([np.sin(np.radians(angle)) for angle in list_diff_yz])
        R_yz = (1/n) * np.sqrt(cosComponent**2 + sinComponent**2)
        
        # Compute MRL for ZX plane
        cosComponent = np.sum([np.cos(np.radians(angle)) for angle in list_diff_zx])
        sinComponent = np.sum([np.sin(np.radians(angle)) for angle in list_diff_zx])
        R_zx = (1/n) * np.sqrt(cosComponent**2 + sinComponent**2)
        
        return [R_xy, R_yz, R_zx]

    @staticmethod
    def _filter_subhalos_for_group(args):
        """Helper function to filter subhalos for a single galaxy group."""
        from myproject.utilities.Subhalo import Subhalo
        from myproject.utilities.GalaxyGroup import GalaxyGroup
        import numpy as np
        
        galaxyGroup, minGGMass, maxGGMass, minSatStellarMass, maxSatStellarMass, minHalfMassRad_kpc, maxHalfMassRad_kpc, centralPosTolerance_kpc = args
        
        if minGGMass is not None and galaxyGroup.getMCrit200() < minGGMass:
            return None
        if maxGGMass is not None and galaxyGroup.getMCrit200() > maxGGMass:
            return None
        
        central_pos = galaxyGroup.getPos()
        galaxyGroupCM = galaxyGroup.getPosCM()
        
        distance = np.linalg.norm(central_pos - galaxyGroupCM)
        if distance > centralPosTolerance_kpc:
            return None  # Signal to skip this group
        
        filtered_subhalos = []
        for subhalo in galaxyGroup.getSubhalos():
            if subhalo.getFlag() == 0:
                continue
            if minSatStellarMass is not None and subhalo.getStellarMass() < minSatStellarMass:
                continue
            if maxSatStellarMass is not None and subhalo.getStellarMass() > maxSatStellarMass:
                continue
            if minHalfMassRad_kpc is not None and subhalo.getHalfMassRad() < minHalfMassRad_kpc:
                continue
            if maxHalfMassRad_kpc is not None and subhalo.getHalfMassRad() > maxHalfMassRad_kpc:
                continue
            filtered_subhalos.append(subhalo)
        
        if len(filtered_subhalos) == 0:
            return None
        
        filtered_galaxyGroup = GalaxyGroup(
            galaxyGroup.getGroupID(), 
            galaxyGroup.getMCrit200(), 
            galaxyGroup.getPosCM(), 
            galaxyGroup.getPos(), 
            filtered_subhalos
        )
        return filtered_galaxyGroup

    @staticmethod
    def _correct_positions_for_group(args):
        """Helper function to correct positions for a single galaxy group."""
        from myproject.utilities.Subhalo import Subhalo
        import numpy as np
        
        galaxyGroup, boxsize = args
        
        central_pos = galaxyGroup.getPos()
        galaxyGroupCM = galaxyGroup.getPosCM()
        
        # Correct group CM position
        corrected_galaxyGroupCM = np.zeros(3)
        for dim in range(3):
            delta = galaxyGroupCM[dim] - central_pos[dim]
            if delta > boxsize / 2:
                corrected_coord = galaxyGroupCM[dim] - boxsize
            elif delta < -boxsize / 2:
                corrected_coord = galaxyGroupCM[dim] + boxsize
            else:
                corrected_coord = galaxyGroupCM[dim]
            corrected_galaxyGroupCM[dim] = corrected_coord
        galaxyGroup.setPosCM(corrected_galaxyGroupCM)
        
        # Set central position to origin
        corrected_central_pos = np.zeros(3)
        galaxyGroup.setPos(corrected_central_pos)
        
        # Correct all subhalo positions
        for subhalo in galaxyGroup.getSubhalos():
            position = subhalo.getPosition()
            corrected_position = np.zeros(3)
            for dim in range(3):
                delta = position[dim] - central_pos[dim]
                if delta > boxsize / 2:
                    corrected_coord = position[dim] - boxsize
                elif delta < -boxsize / 2:
                    corrected_coord = position[dim] + boxsize
                else:
                    corrected_coord = position[dim]
                corrected_position[dim] = corrected_coord
            subhalo.setPosition(corrected_position)
        
        return galaxyGroup

    @staticmethod
    def _serialize_galaxy_group(args):
        """Helper function to serialize a galaxy group's data for HDF5 writing."""
        galaxyGroup, i = args
        
        group_data = {
            'index': i,
            'group_id': galaxyGroup.getGroupID(),
            'MCrit200': galaxyGroup.getMCrit200(),
            'posCM': galaxyGroup.getPosCM(),
            'pos': galaxyGroup.getPos(),
            'subhalos': []
        }
        
        for j, subhalo in enumerate(galaxyGroup.getSubhalos()):
            subhalo_data = {
                'idx': subhalo.getIdx(),
                'group_id': subhalo.getGroupID(),
                'flag': subhalo.getFlag(),
                'mass': subhalo.getMass(),
                'stellarMass': subhalo.getStellarMass(),
                'groupNumber': subhalo.getGroupNumber(),
                'position': subhalo.getPosition(),
                'halfMassRad': subhalo.getHalfMassRad(),
                'vmaxRadius': subhalo.getVmaxRadius(),
                'luminosities': subhalo.getLuminosities()
            }
            group_data['subhalos'].append(subhalo_data)
        
        return group_data
        