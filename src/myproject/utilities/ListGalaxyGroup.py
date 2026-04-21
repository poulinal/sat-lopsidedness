# ADP 2026
from __future__ import annotations
from myproject.utilities.Subhalo import Subhalo
from myproject.utilities.GalaxyGroup import GalaxyGroup
from .parallelTools import parallel_map, get_optimal_processes
import h5py as h5
import numpy as np
import os
import pickle
from typing import Optional, Callable

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
        Computes all pairwPise polar angle differences between satellite galaxies in each galaxy group.
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
        # self.listGalaxyGroups = listGalaxyGroups
        # self.lenGalaxyGroups = len(self.listGalaxyGroups)
        self.setGalaxyGroups(listGalaxyGroups)
        self.headerInformation = headerInformation
        self.list_pairwise_differences : list[list[float]] = []
        self.MRL_values : list[float] = []

        
    @classmethod
    def from_hdf5(cls, h5file : h5.File):
        instance = cls()
        instance.load_from_hdf5(h5file)
        return instance

    def getHeaderInformation(self):
        return self.headerInformation
        
    def addGalaxyGroup(self, galaxyGroup : GalaxyGroup):
        self.listGalaxyGroups.append(galaxyGroup)
        self.lenGalaxyGroups += 1
    
    def addGalaxyGroups(self, galaxyGroups : list[GalaxyGroup]):
        self.listGalaxyGroups.extend(galaxyGroups)
        self.lenGalaxyGroups += len(galaxyGroups)
        
    def setGalaxyGroups(self, listGalaxyGroups : list[GalaxyGroup]):
        self.listGalaxyGroups = listGalaxyGroups
        self.lenGalaxyGroups = len(self.listGalaxyGroups)
        
    def getNumGalaxyGroups(self):
        return self.lenGalaxyGroups
    
    def getAllGalaxyGroups(self):
        return self.listGalaxyGroups
    
    def getGalaxyGroupI(self, i) -> GalaxyGroup:
        return self.listGalaxyGroups[i]

    def getGalaxyGroupByID(self, group_id : int) -> GalaxyGroup | None:
        for galaxyGroup in self.listGalaxyGroups:
            if galaxyGroup.group_id == group_id:
                return galaxyGroup
        return None

    def getSubhaloByID(self, subhalo_id : int) -> Subhalo | None:
        for galaxyGroup in self.listGalaxyGroups:
            subhalo = galaxyGroup.getSubhaloByID(subhalo_id)
            if subhalo is not None:
                return subhalo
        return None

    def getAverageNumSubhalosPerGalaxyGroup(self) -> float:
        total_subhalos = sum(gg.getNumSubhalos() for gg in self.listGalaxyGroups)
        return total_subhalos / self.lenGalaxyGroups if self.lenGalaxyGroups > 0 else 0.0
    
    def getRangeOfNumSubhalos(self) -> tuple[int, int]:
        if not self.listGalaxyGroups:
            return (0, 0)
        num_subhalos_list = [len(gg.getSatelliteSubhalos()) for gg in self.listGalaxyGroups]
        return (min(num_subhalos_list), max(num_subhalos_list)), num_subhalos_list
    
    def getListPairwiseDifferences(self) -> list[list[tuple[float, float, float]]]:
        return self.list_pairwise_differences
            
    def compute_probablity_distribution_of_polar_differences(self, parallelize : bool=False, n_processes : Optional[int]=None, tempSaveDir : str=None, rewrite: bool=False) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        '''
        Docstring for compute_probablity_distribution_of_polar_differences
        Computes all pairwise polar angle differences (in each plane: XY, YZ, ZX) between satellite galaxies in each galaxy group with resepect to the host galaxy.
        Returns a list of all pairwise polar angle differences. Indicies go from 0 to 180º. Indicie of 0 means aligned, 180º means anti-aligned. Values correspond to the density/probability of finding satellite galaxies at a given polar angle difference.
        
        Note: Each entry in the returned list corresponds to a galaxy group, containing a list of tuples. Each tuple contains the polar angle differences (in degrees) between a pair of satellite galaxies in the XY, YZ, and ZX planes respectively. NOT the average of the three planes.
        
        :param self: Description
        :return: Returns a list where each indicie corresponds to a galaxy group, each containing a list of pairwise polar angle differences between satellite galaxies.
        :rtype: list[list[float]]
        '''
        self.list_pairwise_differences = []
        print(f"Total Galaxy Groups to process: {len(self.listGalaxyGroups)} with pairs : {sum([gg.getNumSubhalos() * (gg.getNumSubhalos() - 1) // 2 for gg in self.listGalaxyGroups])}")
        

        
        if parallelize:
            if n_processes is None:
                n_processes = get_optimal_processes(len(self.listGalaxyGroups))
        
            print(f"Computing pairwise differences in parallel with {n_processes} processes...")
            total = len(self.listGalaxyGroups)
            
            import multiprocessing as mp
            with mp.Pool(processes=n_processes) as pool:
                for i, result in enumerate(pool.imap(ListGalaxyGroup._compute_pairwise_for_group, self.listGalaxyGroups), 1):
                    galaxyGroup = self.listGalaxyGroups[i-1]
                    galaxyGroup.setPolarAngleValue(result)
                    self.list_pairwise_differences.append(result)
                    percent = (i / total) * 100
                    print(f"\rProgress: {i}/{total} ({percent:.1f}%)", end='', flush=True)
                print()  # New line after progress
        else:
            start_index = 0
            if tempSaveDir is not None:
                os.makedirs(tempSaveDir, exist_ok=True)
                batch_list_pairwise_differences = []
                
                #check existing temp files to resume
                existing_files = [f for f in os.listdir(tempSaveDir) if f.startswith("pairwise_differences_") and f.endswith(".hdf5")]
                if existing_files and not rewrite:
                    #get the index of each file
                    existing_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
                    last_file = existing_files[-1]
                    print(f"Resuming from existing temp file: {last_file}")
                    start_index = int(last_file.split('_')[2].split('.')[0])
                
            for i, galaxyGroup in enumerate(self.listGalaxyGroups, 1):
                if rewrite == False and tempSaveDir is not None and i <= start_index:
                    continue  # Skip already processed groups
                print(f"\rProgress: Processing Galaxy Group ID {galaxyGroup.getGroupID()}, {i} / {len(self.listGalaxyGroups)}", end='', flush=True)
                group_pairwise_differences = []
                group_pairwise_differences = ListGalaxyGroup._compute_pairwise_for_group(galaxyGroup)
                galaxyGroup.setPolarAngleValue(group_pairwise_differences)

                if type(group_pairwise_differences) != list: #list[tuple[float, float, float]]
                    print(f"WARNING returning.... {group_pairwise_differences, type(group_pairwise_differences)}")
                    return

                if tempSaveDir is not None:
                    batch_list_pairwise_differences.append(group_pairwise_differences)
                else:
                    #flatten
                    flatten_group_pairwise_differences = []
                    for angle in group_pairwise_differences:
                        if isinstance(angle, tuple):
                            flatten_group_pairwise_differences.extend(angle)  # Flatten the tuple
                        else:
                            print(f"Warning: Expected a tuple but got {type(angle)}. Appending as is.")
                            return
                            # flatten_group_pairwise_differences.append(angle)  # Append the float
                    self.list_pairwise_differences.append(flatten_group_pairwise_differences)
                
                #temp save after every 100 groups
                if i % 100 == 0 and tempSaveDir is not None:
                    print(f"\nIntermediate save after processing {i} galaxy groups.")
                    temp_save_path = os.path.join(tempSaveDir, f"pairwise_differences_{i}.hdf5")
                    with h5.File(temp_save_path, 'w') as f:
                        grp = f.create_group('PairwiseDifferences')
                        for j, group_data in enumerate(batch_list_pairwise_differences):
                    #         group_grp = grp.create_group(f'GalaxyGroup_{j}')
                    #         # Store each pairwise difference as a dataset
                    #         for k, pair in enumerate(group_data):
                    #             group_grp.create_dataset(f'Pair_{k}', data=np.array(pair))
                    
                    # save as flattened array to save space
                            group_grp = grp.create_group(f'GalaxyGroup_{j}')
                            # Store each pairwise difference as a single flattened dataset
                            flattened_data = np.array([angle for pair in group_data for angle in pair])
                            group_grp.create_dataset('PairwiseDifferences', data=flattened_data)
                    batch_list_pairwise_differences = []
                        
            # Final save after all groups processed
            if tempSaveDir is not None:
                print(f"\nFinal save after processing all galaxy groups.")
                temp_save_path = os.path.join(tempSaveDir, f"pairwise_differences_{self.getNumGalaxyGroups()}.hdf5")
                with h5.File(temp_save_path, 'w') as f:
                    grp = f.create_group('PairwiseDifferences')
                    for j, group_data in enumerate(batch_list_pairwise_differences):
                        group_grp = grp.create_group(f'GalaxyGroup_{j}')
                        # # Store each pairwise difference as a dataset
                        # for k, pair in enumerate(group_data):
                        #     group_grp.create_dataset(f'Pair_{k}', data=np.array(pair))
                        
                        # save as flattened array to save space
                        flattened_data = np.array([angle for pair in group_data for angle in pair])
                        group_grp.create_dataset('PairwiseDifferences', data=flattened_data)
                    batch_list_pairwise_differences = []
                    
            # accumulate results based on all saved batches
            if tempSaveDir is not None:
                print("Accumulating results from saved batches...")
                self.list_pairwise_differences = []
                for filename in os.listdir(tempSaveDir):
                    if filename.startswith("pairwise_differences_") and filename.endswith(".hdf5"):
                        file_path = os.path.join(tempSaveDir, filename)
                        with h5.File(file_path, 'r') as f:
                            grp = f['PairwiseDifferences']
                            for group_name in grp:
                                # group_data = []
                                # group_grp = grp[group_name]
                                # for pair_name in group_grp:
                                #     pair_data = group_grp[pair_name][()]
                                #     group_data.append(pair_data)
                                # self.list_pairwise_differences.append(group_data)
                                
                                #from flattened save
                                group_grp = grp[group_name]
                                flattened_data = group_grp['PairwiseDifferences'][:]
                                
                                # # Reconstruct original list of tuples from flattened data if needed
                                # group_data = []
                                # for i in range(0, len(flattened_data), 3):
                                #     group_data.append((flattened_data[i], flattened_data[i+1], flattened_data[i+2]))
                                # self.list_pairwise_differences.append(group_data)
                                
                                # reconstruct flattened list
                                group_data = []
                                for angle in flattened_data:
                                    group_data.append(angle)
                                self.list_pairwise_differences.append(group_data)
        
        flatten_list_pairwise_differences = []
        for group_data in self.list_pairwise_differences:
            if isinstance(group_data, list):
                flatten_list_pairwise_differences.extend(group_data)  # Flatten the list of lists
            else:
                print(f"Warning: Expected a list but got {type(group_data)}. Appending as is.")
                return
                # flatten_list_pairwise_differences.append(group_data)  # Append the float
        return flatten_list_pairwise_differences
        
    def compute_probablity_distribution_of_MRL_directionality(self, parallelize: bool = False, n_processes: Optional[int] = None, tempSaveDir : str=None, rewrite: bool=False) -> list[float]:
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
            
        if parallelize:
            if n_processes is None:
                n_processes = get_optimal_processes(len(self.listGalaxyGroups))
            
            print(f"Computing MRL directionality in parallel with {n_processes} processes...")
            total = len(self.listGalaxyGroups)
            
            import multiprocessing as mp
            with mp.Pool(processes=n_processes) as pool:
                results = []
                for i, result in enumerate(pool.imap(ListGalaxyGroup._compute_MRL_for_group, self.listGalaxyGroups), 1):
                    galaxyGroup = self.listGalaxyGroups[i-1]
                    galaxyGroup.setMRLValue(result)
                    results.append(result)
                    percent = (i / total) * 100
                    print(f"\rProgress: {i}/{total} ({percent:.1f}%)", end='', flush=True)
                print()  # New line after progress
            
            # Flatten results (each result is [R_xy, R_yz, R_zx])
            self.MRL_values = []
            for mrl_vals in results:
                self.MRL_values.extend(mrl_vals)
        else:
            if tempSaveDir is not None:
                os.makedirs(tempSaveDir, exist_ok=True)
                batch_MRL_values = []
                #check existing temp files to resume
                existing_files = [f for f in os.listdir(tempSaveDir) if f.startswith("MRL_values_") and f.endswith(".hdf5")]
                if existing_files and not rewrite:
                    #get the index of each file
                    existing_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
                    last_file = existing_files[-1]
                    print(f"Resuming from existing temp file: {last_file}")
                    start_index = int(last_file.split('_')[2].split('.')[0])
            
            for i, galaxyGroup in enumerate(self.listGalaxyGroups, 1):
                if rewrite == False and tempSaveDir is not None and 'start_index' in locals() and i <= start_index:
                    continue  # Skip already processed groups
                    
                print(f"Progress: Processing Galaxy Group {i} / {len(self.listGalaxyGroups)} with {galaxyGroup.getNumSubhalos()} satellites", end='\r', flush=True)
                
                R_values = ListGalaxyGroup._compute_MRL_for_group(galaxyGroup)
                galaxyGroup.setMRLValue(R_values)
                
                if tempSaveDir is not None:
                    batch_MRL_values.extend(R_values)
                else:
                    self.MRL_values.extend(R_values)
                    
                #temp save after every 100 groups
                if tempSaveDir is not None and i % 100 == 0:
                    print(f"\nIntermediate save after processing {i} galaxy groups.")
                    temp_save_path = os.path.join(tempSaveDir, f"MRL_values_{i}.hdf5")
                    with h5.File(temp_save_path, 'w') as f:
                        dset = f.create_dataset('MRL_values', data=np.array(batch_MRL_values))
                    batch_MRL_values = []
            # Final save after all groups processed
            if tempSaveDir is not None:
                print(f"\nFinal save after processing all galaxy groups.")
                temp_save_path = os.path.join(tempSaveDir, f"MRL_values_{self.getNumGalaxyGroups()}.hdf5")
                with h5.File(temp_save_path, 'w') as f:
                    dset = f.create_dataset('MRL_values', data=np.array(batch_MRL_values))
                batch_MRL_values = []
            # accumulate results based on all saved batches
            if tempSaveDir is not None:
                self.MRL_values = []
                for filename in os.listdir(tempSaveDir):
                    if filename.startswith("MRL_values_") and filename.endswith(".hdf5"):
                        file_path = os.path.join(tempSaveDir, filename)
                        with h5.File(file_path, 'r') as f:
                            batch_data = f['MRL_values'][:]
                            self.MRL_values.extend(batch_data)
            
        return self.MRL_values
    
    def compute_MRL_random_distribution_curves_for_LGG(self, num_samples: int = 10000, parallelize: bool = False, n_processes: Optional[int] = None, tempSaveDir: Optional[str] = None, rewrite: bool = False) -> list[list[float]]:
        '''
        Docstring for compute_MRL_random_distribution_curves_for_LGG. Computes the distribution of MRL values for random samples of satellite galaxies to compare against the observed MRL distribution from the galaxy groups. This can help determine if the observed MRL values are significantly different from what would be expected from random distributions of satellites.
        
        :param self: Description
        :rtype: list[float]
        '''
        random_MRL_values = []
        for i, galaxyGroup in enumerate(self.listGalaxyGroups, 1):
            print(f"Progress: Processing random MRL distribution, samples:{num_samples}, for Galaxy Group {i} / {len(self.listGalaxyGroups)} with {galaxyGroup.getNumSubhalos()} satellites", end='\r', flush=True)
            random_MRL_value = ListGalaxyGroup.compute_an_MRL_distribution_curves(num_samples=num_samples, num_non_centrals=len(galaxyGroup.getSatelliteSubhalos()), parallelize=parallelize, n_processes=n_processes, tempSaveDir=tempSaveDir, rewrite=rewrite)
            galaxyGroup.setRandomMRLValue(random_MRL_value)
            random_MRL_values.append(random_MRL_value)
        return random_MRL_values

    def getFilterSubhalos(self, minGGMass : float=None, maxGGMass : float=None, minSatStellarMass : float=None, maxSatStellarMass : float=None, minHalfMassRad_kpc : float=None, maxHalfMassRad_kpc : float=None, centralPosTolerance_kpc : Optional[float]=None, M_r_min : float=None, M_r_max : float=None, M_default_r_min : float=None, M_default_r_max : float=None, satWithinR200 : bool = False, redGalaxies : bool = False, blueGalaxies : bool = False, redDefaultGalaxies : bool = False, blueDefaultGalaxies : bool = False, redBluePoint: float=0.65, minNumGalaxies : Optional[int]=None, maxNumGalaxies : Optional[int]=None, withinXPercentR200 : tuple[float, float] = None, centralIsMostMassive : Optional[bool]=None, parallelize : bool=False, n_processes: Optional[int]=None) -> ListGalaxyGroup:
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
        :param M_r_min: Minimum r-band magnitude to retain a subhalo (default is None)
        :param M_r_max: Maximum r-band magnitude to retain a subhalo (default is None)
        :param satWithinR200: Whether to retain only satellites within R200 (default is False)
        :param redGalaxies: Whether to retain only red galaxies via (g-r) ≥ 0.65 @ z=0 (default is False)
        :param blueGalaxies: Whether to retain only blue galaxies via (g-r) < 0.65 @ z=0 (default is False)
        :param redBluePoint: The (g-r) color cut point to use for separating red and blue galaxies if redGalaxies or blueGalaxies is True (default is None, which uses 0.65)
        :param minNumGalaxies: Minimum number of satellite galaxies required in a galaxy group to retain it (default is None)
        :param maxNumGalaxies: Maximum number of satellite galaxies allowed in a galaxy group to retain it (default is None)
        :param withinXPercentR200: Tuple specifying the range (min, max) as a fraction of R200 within which to retain satellites (default is None)
        :param centralIsMostMassive: Whether to retain only subhalos where the central is the most massive (default is None)
        :param parallelize: Whether to parallelize the filtering process (default is False)
        :param n_processes: Number of processes to use if parallelizing (default is None, which uses optimal number)
        '''
        list_filtered_galaxy_groups : list[GalaxyGroup]= []
        # total = len(self.listGalaxyGroups)
            
        # Prepare arguments for parallel processing
        args_list = [
            (gg, minGGMass, maxGGMass, minSatStellarMass, maxSatStellarMass, minHalfMassRad_kpc, maxHalfMassRad_kpc, centralPosTolerance_kpc, M_r_min, M_r_max, M_default_r_min, M_default_r_max, satWithinR200, redGalaxies, blueGalaxies, redDefaultGalaxies, blueDefaultGalaxies, redBluePoint, minNumGalaxies, maxNumGalaxies, withinXPercentR200, centralIsMostMassive)
            for gg in self.listGalaxyGroups
        ]
        total = len(args_list)
        # print(f"total to process: {total}")
        
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
            # self.setGalaxyGroups([gg for gg in results if gg is not None])
            list_filtered_galaxy_groups = [gg for gg in results if gg is not None]
            
            print(f"After filtering: {self.lenGalaxyGroups} galaxy groups retained.")
        else:
            list_filtered_galaxy_groups : list[GalaxyGroup]= []
            for i, args in enumerate(args_list):
                print(f"Progress: Processing Galaxy Group ID {i+1} / {total}", end='\r')
                list_filtered_galaxy_groups.append(ListGalaxyGroup._filter_subhalos_for_group(args))
                    
            # self.setGalaxyGroups([gg for gg in list_filtered_galaxy_groups if gg is not None])
            list_filtered_galaxy_groups = [gg for gg in list_filtered_galaxy_groups if gg is not None]
            
        # return self.getAllGalaxyGroups()
        return ListGalaxyGroup(listGalaxyGroups=list_filtered_galaxy_groups, headerInformation=self.headerInformation)
                        
    @staticmethod
    def getFilterSubhalosLambda(listGalaxyGroups : ListGalaxyGroup, lambda_func : Callable[[Subhalo], bool], parallelize : bool=False, n_processes: Optional[int]=None) -> ListGalaxyGroup:
        '''
        A more flexible version of getFilterSubhalos that accepts a lambda function to apply custom filtering criteria to each galaxy group. The lambda function should take a GalaxyGroup object as input and return a boolean indicating whether to retain the group (True) or filter it out (False).
        
        :param listGalaxyGroups: The ListGalaxyGroup instance containing the galaxy groups to filter.
        :param lambda_func: A lambda function that defines the filtering criteria. It should accept a GalaxyGroup object and return True to retain or False to filter out.
        :param parallelize: Whether to parallelize the filtering process (default is False).
        :param n_processes: Number of processes to use if parallelizing (default is None, which uses optimal number).
        :return: A new ListGalaxyGroup instance containing only the filtered galaxy groups.
        '''
        list_filtered_galaxy_groups : list[GalaxyGroup]= []
        
        if parallelize:
            if n_processes is None:
                n_processes = get_optimal_processes(len(listGalaxyGroups.listGalaxyGroups))
            
            print(f"Filtering subhalos with custom lambda in parallel with {n_processes} processes...")
            
            import multiprocessing as mp
            with mp.Pool(processes=n_processes) as pool:
                results = []
                for i, result in enumerate(pool.imap(lambda gg: ListGalaxyGroup._filter_subhalos_with_lambda(gg, lambda_func), listGalaxyGroups.listGalaxyGroups), 1):
                    results.append(result)
                    percent = (i / len(listGalaxyGroups.listGalaxyGroups)) * 100
                    print(f"\rProgress: {i}/{len(listGalaxyGroups.listGalaxyGroups)} ({percent:.1f}%)", end='', flush=True)
                print()  # New line after progress
            
            list_filtered_galaxy_groups = [gg for gg in results if gg is not None]
            
            print(f"After filtering with lambda: {len(list_filtered_galaxy_groups)} galaxy groups retained.")
        else:
            for i, galaxyGroup in enumerate(listGalaxyGroups.listGalaxyGroups, 1):
                print(f"Progress: Processing Galaxy Group ID {i} / {len(listGalaxyGroups.listGalaxyGroups)}", end='\r')
                if lambda_func(galaxyGroup):
                    list_filtered_galaxy_groups.append(galaxyGroup)
                    
        return ListGalaxyGroup(listGalaxyGroups=list_filtered_galaxy_groups, headerInformation=listGalaxyGroups.headerInformation)

    def getCorrectedPositions(self, boxsize : float, parallelize : bool=False, n_processes: Optional[int]=None) -> ListGalaxyGroup:
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
            
        else:
            results = []
            for args in args_list:
                print(f"Progress: Processing Galaxy Group ID {args[0].getGroupID()} / {len(self.listGalaxyGroups)}", end='\r')
                results.append(ListGalaxyGroup._correct_positions_for_group(args))
                
        # self.setGalaxyGroups(results)
        # return self.getAllGalaxyGroups()
        # return 
        return ListGalaxyGroup(listGalaxyGroups=results, headerInformation=self.headerInformation)
       
       
       
    def save_to_hdf5(
        self,
        h5file: h5.File,
        overwrite: bool = True,
        parallize: bool = False,
        n_processes: Optional[int] = None,
        storage_layout: str = 'attrs',
        compression: Optional[str] = None,
        compression_opts: Optional[int] = None,):
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
        if storage_layout not in {'attrs', 'datasets'}:
            raise ValueError("storage_layout must be 'attrs' or 'datasets'")

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

            # Record the storage layout for future readers
            h5file.attrs['ListGalaxyGroup_storage_layout'] = storage_layout
            
            total = len(self.listGalaxyGroups)
            
            if parallize and n_processes is None:
                n_processes = get_optimal_processes(len(self.listGalaxyGroups))
            
                print(f"Serializing galaxy group data in parallel with {n_processes} processes...")
            
            # Prepare arguments for parallel processing
            args_list = [(gg, i) for i, gg in enumerate(self.listGalaxyGroups)]

            serializer = (
                ListGalaxyGroup._serialize_galaxy_group
                if storage_layout == 'attrs'
                else ListGalaxyGroup._serialize_galaxy_group_datasets
            )
            
            if parallize:
                # Use imap to get results as they complete (allows progress tracking)
                import multiprocessing as mp
                with mp.Pool(processes=n_processes) as pool:
                    serialized_data = []
                    for i, result in enumerate(pool.imap(serializer, args_list), 1):
                        serialized_data.append(result)
                        percent = (i / total) * 100
                        print(f"\rSerialization: {i}/{total} ({percent:.1f}%)", end='', flush=True)
                    print()  # New line after progress
            else:
                print("Serializing galaxy group data sequentially...")
                serialized_data = []
                for i, args in enumerate(args_list, 1):
                    result = serializer(args)
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
                gg_grp.attrs['RCrit200'] = group_data['RCrit200']
                gg_grp.attrs['MCrit200'] = group_data['MCrit200']
                gg_grp.attrs['posCM'] = group_data['posCM']
                gg_grp.attrs['pos'] = group_data['pos']
                
                subhalos_grp = gg_grp.create_group('Subhalos')

                if storage_layout == 'attrs':
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
                        sh_grp.attrs['luminositiesSDSS'] = subhalo_data['luminositiesSDSS']
                        sh_grp.attrs['joinTimes'] = subhalo_data['joinTimes']
                else:
                    # Bulk datasets (significantly faster than many tiny groups/attrs)
                    for field_name, arr in group_data['subhalos'].items():
                        subhalos_grp.create_dataset(
                            field_name,
                            data=arr,
                            compression=compression,
                            compression_opts=compression_opts,
                        )
            
            print(f"\nCompleted writing {len(serialized_data)} galaxy groups to HDF5.")

    # ---------------------------------------------------------------------
    # Faster HDF5 I/O notes
    # ---------------------------------------------------------------------
    # The current on-disk format (attrs + many small groups) is easy to
    # inspect manually, but it is slow for large catalogs. The biggest
    # bottlenecks are:
    #   1) Creating thousands of HDF5 objects (groups/datasets)
    #   2) Per-subhalo attribute writes/reads
    #   3) In load, repeatedly recomputing central/satellites via addSubhalo()
    #
    # We keep the existing layout for backward compatibility, but the loader
    # is optimized below to avoid O(n^2) behavior.
                    
    def load_from_hdf5(self, h5file : h5.File, parallelize: bool=False, n_processes: Optional[int]=None):
        self.listGalaxyGroups = []
        self.headerInformation = {}
        for key, value in h5file.attrs.items():
            self.headerInformation[key] = value
        
        grp = h5file['GalaxyGroups']
        gg_keys = list(grp.keys())
        total = len(gg_keys)
        
        if parallelize:
            if n_processes is None:
                n_processes = get_optimal_processes(total)
            
            print(f"Loading from HDF5 in parallel with {n_processes} processes...")
            
            # Create filename to pass to worker (HDF5 objects can't be pickled)
            h5_filename = h5file.filename
            
            import multiprocessing as mp
            with mp.Pool(processes=n_processes) as pool:
                args_list = [(h5_filename, gg_key) for gg_key in gg_keys]
                results = []
                for i, result in enumerate(pool.imap(ListGalaxyGroup._load_group_from_hdf5_auto, args_list), 1):
                    results.append(result)
                    percent = (i / total) * 100
                    print(f"\rProgress: {i}/{total} ({percent:.1f}%)", end='', flush=True)
                print()  # New line after progress
            
            self.listGalaxyGroups = results
        else:
            for i, gg_key in enumerate(gg_keys):
                print(f"Progress: {i+1}/{total}", end='\r')
                galaxyGroup = ListGalaxyGroup._load_group_from_hdf5_auto((h5file.filename, gg_key))
                self.listGalaxyGroups.append(galaxyGroup)
        
        self.lenGalaxyGroups = len(self.listGalaxyGroups)
        print(f"\nLoaded {len(self.listGalaxyGroups)} galaxy groups from HDF5.")

    @staticmethod
    def compute_an_MRL_distribution_curves(num_samples: int = 10000, num_non_centrals: int = 20, parallelize: bool = False, n_processes: Optional[int] = None, tempSaveDir: Optional[str] = None, rewrite: bool = False) -> list[tuple[np.ndarray, np.ndarray]]:
        '''
        Docstring for compute_an_MRL_distribution_curves. Plots the distribution of MRL values for random samples of satellite galaxies to compare against the observed MRL distribution from the galaxy groups. This can help determine if the observed MRL values are significantly different from what would be expected from random distributions of satellites.
        
        :param self: Description
        :param num_samples: Description
        :type num_samples: int
        :param num_non_centrals: Description
        :type num_non_centrals: int
        :param parallelize: Description
        :type parallelize: bool
        :param n_processes: Description
        :type n_processes: Optional[int]
        :param tempSaveDir: Description
        :type tempSaveDir: Optional[str]
        :param rewrite: Description
        :type rewrite: bool
        :return: Description
        :rtype: tuple[ndarray, ndarray, ndarray]
        '''
        if num_non_centrals <= 0:
            return []

        random_MRL_values: list[float] = []
        for _ in range(num_samples):
            # random_angles = np.random.uniform(0, 180, size=num_non_centrals)  # Random angles between 0 and 180 degrees
            #for a given random position, get the xy yz and xz angles
            # random_angles_xy = np.radians(np.random.uniform(0, 180, size=num_non_centrals))
            # random_angles_yz = np.radians(np.random.uniform(0, 180, size=num_non_centrals))
            # random_angles_xz = np.radians(np.random.uniform(0, 180, size=num_non_centrals))

            # rel_positions = np.random.uniform(-1, 1, size=(num_non_centrals, 3))  # Random relative positions in 3D space
            # random_angles_xy = np.arctan2(rel_positions[:, 1], rel_positions[:, 0]) #% np.pi # Angle in XY plane
            # random_angles_yz = np.arctan2(rel_positions[:, 2], rel_positions[:, 1]) #% np.pi  # Angle in YZ plane
            # random_angles_xz = np.arctan2(rel_positions[:, 2], rel_positions[:, 0]) #% np.pi  # Angle in XZ plane

            # Sample isotropic directions by drawing random 3D vectors and normalizing.
            # This avoids angle biases introduced by sampling uniformly in a cube.
            vec = np.random.normal(size=(num_non_centrals, 3))
            vec /= np.linalg.norm(vec, axis=1, keepdims=True)

            random_angles_xy = np.arctan2(vec[:, 1], vec[:, 0])
            random_angles_yz = np.arctan2(vec[:, 2], vec[:, 1])
            random_angles_xz = np.arctan2(vec[:, 2], vec[:, 0])
            
            
            cos_sum_xy = np.sum(np.cos(random_angles_xy))
            sin_sum_xy = np.sum(np.sin(random_angles_xy))
            R_xy = np.sqrt(cos_sum_xy**2 + sin_sum_xy**2) / num_non_centrals
            
            cos_sum_yz = np.sum(np.cos(random_angles_yz))
            sin_sum_yz = np.sum(np.sin(random_angles_yz))
            R_yz = np.sqrt(cos_sum_yz**2 + sin_sum_yz**2) / num_non_centrals
            
            cos_sum_xz = np.sum(np.cos(random_angles_xz))
            sin_sum_xz = np.sum(np.sin(random_angles_xz))
            R_xz = np.sqrt(cos_sum_xz**2 + sin_sum_xz**2) / num_non_centrals
            
            random_MRL_values.extend([R_xy, R_yz, R_xz])
            
        return random_MRL_values
    
    @staticmethod
    def get_histogram_bins(values: list[float], bins: Optional[float] = None, binsize: Optional[float] = None, binLow: Optional[float]=None, binHigh: Optional[float]=None, density: bool = False, normalize_to_one: bool = False, errorbarType: str = 'poisson') -> np.ndarray:
        """Helper function to compute histogram bins for pairwise differences.

        New parameter `normalize_to_one` when True normalizes bin heights so their
        sum equals 1 (i.e. discrete probability per bin). This is distinct from
        `density=True` which returns a probability density (integral over x equals 1).
        """
        import numpy as np
        bin = None
        if bins is not None:
            bin = bins
        elif binsize is not None and binLow is not None and binHigh is not None:
            bin = np.arange(binLow, binHigh + binsize, binsize)
        if bin is None:
            bin = 'auto'  # Default to 'auto' if no valid binning parameters provided

        values = np.asarray(values)

        # If user requests discrete normalization to one, compute counts and
        # normalize by the total count so sum(hist) == 1.
        if normalize_to_one:
            counts, bin_edges = np.histogram(values, bins=bin, density=False)
            total = counts.sum()
            if total > 0:
                hist = counts.astype(float) / total
            else:
                hist = np.zeros_like(counts, dtype=float)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            if errorbarType == 'poisson':
                if total > 0:
                    errorbars = np.sqrt(counts) / total
                else:
                    errorbars = np.zeros_like(hist)
            elif errorbarType == 'bootstrap':
                n_bootstrap = 1000
                bootstrap_histograms = []
                for _ in range(n_bootstrap):
                    resampled_values = np.random.choice(values, size=len(values), replace=True)
                    b_counts, _ = np.histogram(resampled_values, bins=bin, density=False)
                    b_total = b_counts.sum()
                    if b_total > 0:
                        bootstrap_histograms.append(b_counts.astype(float) / b_total)
                    else:
                        bootstrap_histograms.append(np.zeros_like(counts, dtype=float))
                errorbars = np.std(bootstrap_histograms, axis=0)
            else:
                raise ValueError("Invalid errorbarType. Choose 'poisson' or 'bootstrap'.")

            return hist, bin_centers, errorbars

        # Otherwise, follow traditional density/count behavior based on `density`
        if density:
            hist, bin_edges = np.histogram(values, bins=bin, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_widths = bin_edges[1:] - bin_edges[:-1]  # need bin widths to properly normalize poisson errors when density=True

            if errorbarType == 'poisson':
                N = len(values)
                if N > 0:
                    errorbars = np.sqrt(hist / (N * bin_widths))  # Poisson errors normalized to density
                else:
                    errorbars = np.zeros_like(hist)
            elif errorbarType == 'bootstrap':
                n_bootstrap = 1000
                bootstrap_histograms = []
                for _ in range(n_bootstrap):
                    resampled_values = np.random.choice(values, size=len(values), replace=True)
                    bootstrap_hist, _ = np.histogram(resampled_values, bins=bin, density=True)
                    bootstrap_histograms.append(bootstrap_hist)
                errorbars = np.std(bootstrap_histograms, axis=0)
            else:
                raise ValueError("Invalid errorbarType. Choose 'poisson' or 'bootstrap'.")

            return hist, bin_centers, errorbars

        # density == False and normalize_to_one == False -> return raw counts
        counts, bin_edges = np.histogram(values, bins=bin, density=False)
        hist = counts.astype(float)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        if errorbarType == 'poisson':
            errorbars = np.sqrt(counts)
        elif errorbarType == 'bootstrap':
            n_bootstrap = 1000
            bootstrap_histograms = []
            for _ in range(n_bootstrap):
                resampled_values = np.random.choice(values, size=len(values), replace=True)
                b_counts, _ = np.histogram(resampled_values, bins=bin, density=False)
                bootstrap_histograms.append(b_counts)
            errorbars = np.std(bootstrap_histograms, axis=0)
        else:
            raise ValueError("Invalid errorbarType. Choose 'poisson' or 'bootstrap'.")

        return hist, bin_centers, errorbars
    
    # Standalone functions for multiprocessing (must be picklable)
    @staticmethod
    def _load_group_from_hdf5(args):
        """Helper function to load a single galaxy group from HDF5 file."""
        import h5py as h5
        from myproject.utilities.Subhalo import Subhalo
        from myproject.utilities.GalaxyGroup import GalaxyGroup
        
        h5_filename, gg_key = args
        
        with h5.File(h5_filename, 'r') as h5file:
            gg_grp = h5file['GalaxyGroups'][gg_key]
            galaxy_group_id = gg_grp.attrs['group_id']
            RCrit200 = gg_grp.attrs['RCrit200']
            MCrit200 = gg_grp.attrs['MCrit200']
            posCM = gg_grp.attrs['posCM']
            pos = gg_grp.attrs['pos']

            # Important performance detail:
            # Avoid GalaxyGroup.addSubhalo() in a loop here.
            # addSubhalo() recomputes central/satellites every append, which is
            # O(n^2) per group. Instead, build the list once, then initialize.
            subhalos: list[Subhalo] = []
            subhalos_grp = gg_grp['Subhalos']
            for sh_key in subhalos_grp:
                sh_grp = subhalos_grp[sh_key]
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
                luminositiesSDSS = sh_grp.attrs['luminositiesSDSS']
                joinTimes = sh_grp.attrs['joinTimes']

                subhalos.append(
                    Subhalo(
                        idx,
                        group_id,
                        flag,
                        mass,
                        stellarMass,
                        groupNumber,
                        position,
                        halfMassRad,
                        vmaxRadius,
                        luminosities,
                        luminositiesSDSS,
                        group_pos=pos,
                        joiningRedshift=joinTimes,
                    )
                )

            galaxyGroup = GalaxyGroup(galaxy_group_id, RCrit200, MCrit200, posCM, pos, listSubhalos=subhalos)
            return galaxyGroup

    @staticmethod
    def _load_group_from_hdf5_auto(args):
        """Auto-detect loader for either 'attrs' or 'datasets' subhalo storage."""
        import h5py as h5

        h5_filename, gg_key = args
        with h5.File(h5_filename, 'r') as h5file:
            gg_grp = h5file['GalaxyGroups'][gg_key]
            if 'Subhalos' in gg_grp and 'idx' in gg_grp['Subhalos']:
                return ListGalaxyGroup._load_group_from_hdf5_datasets(args)
        return ListGalaxyGroup._load_group_from_hdf5(args)

    @staticmethod
    def _load_group_from_hdf5_datasets(args):
        """Load a single galaxy group from the faster dataset-based layout."""
        import h5py as h5
        import numpy as np
        from myproject.utilities.Subhalo import Subhalo
        from myproject.utilities.GalaxyGroup import GalaxyGroup

        h5_filename, gg_key = args
        with h5.File(h5_filename, 'r') as h5file:
            gg_grp = h5file['GalaxyGroups'][gg_key]
            galaxy_group_id = gg_grp.attrs['group_id']
            RCrit200 = gg_grp.attrs['RCrit200']
            MCrit200 = gg_grp.attrs['MCrit200']
            posCM = gg_grp.attrs['posCM']
            pos = gg_grp.attrs['pos']

            sh_grp = gg_grp['Subhalos']
            idx = sh_grp['idx'][:]
            group_id = sh_grp['group_id'][:]
            flag = sh_grp['flag'][:]
            mass = sh_grp['mass'][:]
            stellarMass = sh_grp['stellarMass'][:]
            groupNumber = sh_grp['groupNumber'][:]
            position = sh_grp['position'][:]
            halfMassRad = sh_grp['halfMassRad'][:]
            vmaxRadius = sh_grp['vmaxRadius'][:]
            luminosities = sh_grp['luminosities'][:]
            luminositiesSDSS = sh_grp['luminositiesSDSS'][:]
            joinTimes = sh_grp['joinTimes'][:]

            subhalos: list[Subhalo] = []
            n = len(idx)
            for i in range(n):
                jt_arr = np.asarray(joinTimes[i])
                subhalos.append(
                    Subhalo(
                        int(idx[i]),
                        int(group_id[i]),
                        int(flag[i]),
                        mass[i],
                        float(stellarMass[i]),
                        int(groupNumber[i]),
                        position[i],
                        float(halfMassRad[i]),
                        float(vmaxRadius[i]),
                        luminosities[i],
                        luminositiesSDSS[i],
                        group_pos=pos,
                        joiningRedshift=jt_arr,
                    )
                )

            return GalaxyGroup(galaxy_group_id, RCrit200, MCrit200, posCM, pos, listSubhalos=subhalos)

    @staticmethod
    def _serialize_galaxy_group_datasets(args: tuple[GalaxyGroup, int]):
        """Serialize a galaxy group into bulk NumPy arrays (fast HDF5 datasets)."""
        import numpy as np

        galaxyGroup, i = args
        subhalos = list(galaxyGroup.getSubhalos())
        n = len(subhalos)

        def _as_1d_array(x):
            arr = np.asarray(x)
            return arr.ravel() if arr.shape != () else arr.reshape(1)

        if n == 0:
            mass_len = 6
            lum_len = 8
            join_len = 15
        else:
            mass_len = _as_1d_array(subhalos[0].getMass()).size
            lum_len = _as_1d_array(subhalos[0].getLuminosities()).size
            join_len = 1
            for sh in subhalos:
                jt = sh.getJoiningRedshiftInfo()
                if jt is np.nan:
                    continue
                try:
                    arr = _as_1d_array(jt)
                except Exception:
                    continue
                join_len = max(join_len, arr.size)

        idx = np.empty(n, dtype=np.int64)
        group_id = np.empty(n, dtype=np.int64)
        flag = np.empty(n, dtype=np.int16)
        stellarMass = np.empty(n, dtype=np.float64)
        groupNumber = np.empty(n, dtype=np.int64)
        position = np.empty((n, 3), dtype=np.float64)
        halfMassRad = np.empty(n, dtype=np.float64)
        vmaxRadius = np.empty(n, dtype=np.float64)

        mass = np.empty((n, mass_len), dtype=np.float64)
        luminosities = np.empty((n, lum_len), dtype=np.float64)
        luminositiesSDSS = np.empty((n, lum_len), dtype=np.float64)
        joinTimes = np.full((n, join_len), np.nan, dtype=np.float64)

        for j, sh in enumerate(subhalos):
            idx[j] = sh.getIdx()
            group_id[j] = sh.getGroupID()
            flag[j] = sh.getFlag()
            stellarMass[j] = sh.getStellarMass()
            groupNumber[j] = sh.getGroupNumber()
            position[j] = sh.getPosition()
            halfMassRad[j] = sh.getHalfMassRad()
            vmaxRadius[j] = sh.getVmaxRadius()

            m = _as_1d_array(sh.getMass())
            if m.size != mass_len:
                raise ValueError(
                    f"Inconsistent mass vector length in group {galaxyGroup.getGroupID()}: expected {mass_len}, got {m.size}"
                )
            mass[j] = m

            lum = _as_1d_array(sh.getLuminosities())
            if lum.size != lum_len:
                raise ValueError(
                    f"Inconsistent luminosities length in group {galaxyGroup.getGroupID()}: expected {lum_len}, got {lum.size}"
                )
            luminosities[j] = lum

            lum_sdss = _as_1d_array(sh.getLuminositiesSDSS())
            if lum_sdss.size != lum_len:
                raise ValueError(
                    f"Inconsistent SDSS luminosities length in group {galaxyGroup.getGroupID()}: expected {lum_len}, got {lum_sdss.size}"
                )
            luminositiesSDSS[j] = lum_sdss

            jt = sh.getJoiningRedshiftInfo()
            if jt is not np.nan:
                jt_arr = _as_1d_array(jt)
                joinTimes[j, : jt_arr.size] = jt_arr

        group_data = {
            'index': i,
            'group_id': galaxyGroup.getGroupID(),
            'RCrit200': galaxyGroup.getRCrit200(),
            'MCrit200': galaxyGroup.getMCrit200(),
            'posCM': galaxyGroup.getPosCM(),
            'pos': galaxyGroup.getPos(),
            'subhalos': {
                'idx': idx,
                'group_id': group_id,
                'flag': flag,
                'mass': mass,
                'stellarMass': stellarMass,
                'groupNumber': groupNumber,
                'position': position,
                'halfMassRad': halfMassRad,
                'vmaxRadius': vmaxRadius,
                'luminosities': luminosities,
                'luminositiesSDSS': luminositiesSDSS,
                'joinTimes': joinTimes,
            },
        }

        return group_data

    @staticmethod
    def _compute_pairwise_for_group(galaxyGroup : GalaxyGroup, projections: list[str]=['xy', 'yz', 'xz']) -> list[tuple[float, float, float]]:
        """Helper function to compute pairwise differences for a single galaxy group.
        
        Memory-efficient streaming approach: computes angles upfront (minimal memory),
        then iterates through pairs without storing full N×N matrices.
        """
        import numpy as np
        
        subhalos = galaxyGroup.getSatelliteSubhalos()
        num_subhalos = len(subhalos)
        if num_subhalos < 2:
            return []

        central_pos = galaxyGroup.getCentralSubhalo().getPosition()
        
        # Vectorize angle computation (minimal memory footprint)
        positions = np.array([sh.getPosition() for sh in subhalos], dtype=np.float32)
        rel_pos = positions - central_pos
    
        
        if 'xy' in projections:
            angles_xy = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])
        if 'yz' in projections:
            angles_yz = np.arctan2(rel_pos[:, 2], rel_pos[:, 1])
        if 'xz' in projections:
            angles_xz = np.arctan2(rel_pos[:, 2], rel_pos[:, 0])

        # Stream-compute pairwise differences without allocating full matrices
        group_pairwise_differences = []
        deg = 180.0 / np.pi
        
        for i in range(num_subhalos):
            # print(f"\rProcessing Subhalo {i+1}/{num_subhalos} in Galaxy Group ID {galaxyGroup.getGroupID()} with total pairs {num_subhalos * (num_subhalos - 1) // 2}", end='', flush=True)
            for j in range(i + 1, num_subhalos):
                # Compute angle differences for each plane (in radians, then convert)
                # where delta_xy=0 corresponds to the same side and delta_xy=180 corresponds to opposite sides
                if 'xy' in projections:
                    delta_xy = angles_xy[i] - angles_xy[j]
                if 'yz' in projections:
                    delta_yz = angles_yz[i] - angles_yz[j]
                if 'xz' in projections:
                    delta_xz = angles_xz[i] - angles_xz[j]

                # Normalize differences to [0, π]
                if 'xy' in projections:
                    diff_xy = np.abs((delta_xy + np.pi) % (2 * np.pi) - np.pi)
                else:
                    diff_xy = np.nan
                if 'yz' in projections:
                    diff_yz = np.abs((delta_yz + np.pi) % (2 * np.pi) - np.pi)
                else:
                    diff_yz = np.nan
                if 'xz' in projections:
                    diff_xz = np.abs((delta_xz + np.pi) % (2 * np.pi) - np.pi)
                else:
                    diff_xz = np.nan

                # Convert to degrees
                pairwise_difference = (diff_xy * deg, diff_yz * deg, diff_xz * deg)
                
                # Append only non-NaN values
                filtered_difference = tuple(val for val in pairwise_difference if not np.isnan(val))
                if filtered_difference:
                    group_pairwise_differences.append(filtered_difference)
        
        return group_pairwise_differences
    
    @staticmethod
    def _compute_MRL_for_group(galaxyGroup : GalaxyGroup, projections : list[str]=['xy', 'yz', 'xz']) -> list[float]:
        """Helper function to compute MRL for a single galaxy group using individual satellite angles.
        
        Computes the Mean Resultant Length for each plane (XY, YZ, XZ) based on the angular
        distribution of satellites relative to the central galaxy.
        """
        import numpy as np
        
        subhalos = galaxyGroup.getSatelliteSubhalos()
        id = galaxyGroup.getGroupID()
        n = len(subhalos)
        
        if n == 0:
            return [0.0, 0.0, 0.0]
        
        central_pos = galaxyGroup.getCentralSubhalo().getPosition()
        
        # Vectorize: Get all satellite positions at once
        positions = np.array([sh.getPosition() for sh in subhalos], dtype=np.float32)
        rel_pos = positions - central_pos
        
        # Compute polar angles using arctan
        if 'xy' in projections:
            angles_xy = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])  # XY plane: arctan2(y, x)
        if 'yz' in projections:
            angles_yz = np.arctan2(rel_pos[:, 2], rel_pos[:, 1])  # YZ plane: arctan2(z, y)
        if 'xz' in projections:
            angles_xz = np.arctan2(rel_pos[:, 2], rel_pos[:, 0])  # XZ plane: arctan2(z, x)
        
        # Compute MRL for XY plane
        if 'xy' in projections:
            cos_sum_xy = np.sum(np.cos(angles_xy))
            sin_sum_xy = np.sum(np.sin(angles_xy))
            R_xy = (1/n) * np.sqrt(cos_sum_xy**2 + sin_sum_xy**2)
        else:
            R_xy = np.nan

        # Compute MRL for YZ plane
        if 'yz' in projections:
            cos_sum_yz = np.sum(np.cos(angles_yz))
            sin_sum_yz = np.sum(np.sin(angles_yz))
            R_yz = (1/n) * np.sqrt(cos_sum_yz**2 + sin_sum_yz**2)
        else:
            R_yz = np.nan

        # Compute MRL for XZ plane
        if 'xz' in projections:
            cos_sum_xz = np.sum(np.cos(angles_xz))
            sin_sum_xz = np.sum(np.sin(angles_xz))
            R_xz = (1/n) * np.sqrt(cos_sum_xz**2 + sin_sum_xz**2)
        else:
            R_xz = np.nan

        # print(f"final: {[R_xy, R_yz, R_xz]}") if id == 0 else None
        
        return [R_xy, R_yz, R_xz]

    @staticmethod
    def _filter_subhalos_for_group(args : tuple[GalaxyGroup, float, float, float, float, float, float, float, float, float,float, float, float, bool, bool, bool, bool, bool, float, int, int, tuple[float, float]]):
        """Helper function to filter subhalos for a single galaxy group."""
        from myproject.utilities.Subhalo import Subhalo
        from myproject.utilities.GalaxyGroup import GalaxyGroup
        import numpy as np
        
        galaxyGroup, minGGMass, maxGGMass, minSatStellarMass, maxSatStellarMass, minHalfMassRad_kpc, maxHalfMassRad_kpc, centralPosTolerance_kpc, M_r_min, M_r_max, M_default_r_min, M_default_r_max, satWithinR200, redGalaxies, blueGalaxies, redDefaultGalaxies, blueDefaultGalaxies, redBluePoint, minNumGalaxies, maxNumGalaxies, withinXPercentR200, centralIsMostMassive = args

        # Quick group-level checks (unchanged)
        if minGGMass is not None and galaxyGroup.getMCrit200() <= minGGMass:
            return None
        if maxGGMass is not None and galaxyGroup.getMCrit200() >= maxGGMass:
            return None

        if centralIsMostMassive is not None:
            central = galaxyGroup.getCentralSubhalo()
            most_massive = galaxyGroup.getMostMassiveSubhalo()
            if centralIsMostMassive and central != most_massive:
                return None
            if not centralIsMostMassive and central == most_massive:
                return None

        central_pos = galaxyGroup.getCentralSubhalo().getPosition()
        galaxyGroupCM = galaxyGroup.getPosCM()

        distance = np.linalg.norm(central_pos - galaxyGroupCM)
        if centralPosTolerance_kpc is not None and distance > centralPosTolerance_kpc:
            return None  # Skip this group

        # Vectorize per-subhalo checks using NumPy arrays to avoid Python loops
        subhalos = list(galaxyGroup.getSubhalos())
        if len(subhalos) == 0:
            return None

        # Extract arrays of attributes
        flags = np.array([sh.getFlag() for sh in subhalos])
        stellar = np.array([sh.getStellarMass() for sh in subhalos])
        halfrad = np.array([sh.getHalfMassRad() for sh in subhalos])
        positions = np.array([sh.getPosition() for sh in subhalos])
        try:
            r_mag = np.array([sh.getRbandMagnitude() for sh in subhalos], dtype=float)
        except Exception:
            r_mag = np.array([np.nan for _ in subhalos], dtype=float)
        try:
            def_r_mag = np.array([sh.getDefaultRbandMagnitude() for sh in subhalos], dtype=float)
            # print(f"len def_r_mag: {len(def_r_mag)}")
        except Exception:
            def_r_mag = np.array([np.nan for _ in subhalos], dtype=float)
        try:
            g_mag = np.array([sh.getGbandMagnitude() for sh in subhalos], dtype=float)
        except Exception:
            g_mag = np.array([np.nan for _ in subhalos], dtype=float)
        try:
            def_g_mag = np.array([sh.getDefaultGbandMagnitude() for sh in subhalos], dtype=float)
            # print(f"len def_g_mag: {len(def_g_mag)}")
        except Exception:
            def_g_mag = np.array([np.nan for _ in subhalos], dtype=float)

        is_central = np.array([sh is galaxyGroup.getCentralSubhalo() for sh in subhalos])

        # Start with flag mask
        mask = (flags == 1)

        # Stellar mass filters (apply to all subhalos, including central)
        if minSatStellarMass is not None:
            mask &= (stellar >= minSatStellarMass)
        if maxSatStellarMass is not None:
            mask &= (stellar <= maxSatStellarMass)

        # Half-mass radius filters
        if minHalfMassRad_kpc is not None:
            mask &= (halfrad >= minHalfMassRad_kpc)
        if maxHalfMassRad_kpc is not None:
            mask &= (halfrad <= maxHalfMassRad_kpc)

        # r-band magnitude filters: original logic skipped subhalo if NaN OR out-of-range.
        if M_r_min is not None:
            valid = ~np.isnan(r_mag)
            mask &= (valid & (r_mag > M_r_min))
        if M_r_max is not None:
            valid = ~np.isnan(r_mag)
            mask &= (valid & (r_mag < M_r_max))

        # default r-band magnitude
        if M_default_r_min is not None:
            valid = ~np.isnan(def_r_mag)
            mask &= (valid & (def_r_mag > M_default_r_min))
        if M_default_r_max is not None:
            valid = ~np.isnan(def_r_mag)
            mask &= (valid & (def_r_mag < M_default_r_max))

        # Distance-based checks (apply only to non-central entries)
        if satWithinR200 or withinXPercentR200 is not None:
            distances = np.linalg.norm(positions - central_pos, axis=1)
            r200 = galaxyGroup.getRCrit200()
            if satWithinR200:
                # central is always allowed; non-central must be within r200
                mask &= (is_central | (distances <= r200))
            if withinXPercentR200 is not None:
                min_radius = withinXPercentR200[0] * r200
                max_radius = withinXPercentR200[1] * r200
                mask &= (is_central | ((distances >= min_radius) & (distances <= max_radius)))

        # Color selection (red/blue) - only applies to non-central
        if redGalaxies or blueGalaxies:
            color_valid = ~np.isnan(g_mag) & ~np.isnan(r_mag)
            g_r_color = (g_mag - r_mag) * -1
            if redGalaxies:
                mask &= (is_central | (color_valid & (g_r_color >= redBluePoint)))
            if blueGalaxies:
                mask &= (is_central | (color_valid & (g_r_color < redBluePoint)))

        if redDefaultGalaxies or blueDefaultGalaxies:
            color_valid = ~np.isnan(def_g_mag) & ~np.isnan(def_r_mag)
            g_r_color = (def_g_mag - def_r_mag) #* -1
            if redDefaultGalaxies:
                mask &= (is_central | (color_valid & (g_r_color >= redBluePoint)))
            if blueDefaultGalaxies:
                mask &= (is_central | (color_valid & (g_r_color < redBluePoint)))

        # Build filtered subhalo list preserving original object references
        filtered_subhalos = [sh for sh, m in zip(subhalos, mask) if m]

        if len(filtered_subhalos) <= 1:  # need at least central + one satellite
            return None

        if minNumGalaxies is not None and len(filtered_subhalos) < minNumGalaxies:
            return None
        if maxNumGalaxies is not None and len(filtered_subhalos) > maxNumGalaxies:
            return None

        filtered_galaxyGroup = GalaxyGroup(
            galaxyGroup.getGroupID(),
            galaxyGroup.getRCrit200(),
            galaxyGroup.getMCrit200(),
            galaxyGroup.getPosCM(),
            galaxyGroup.getPos(),
            filtered_subhalos,
        )
        return filtered_galaxyGroup

    @staticmethod
    def _filter_subhalos_with_lambda(args : tuple[GalaxyGroup, Callable[[Subhalo], bool]]):
        """Helper function to filter subhalos for a single galaxy group using a custom lambda function."""
        from myproject.utilities.Subhalo import Subhalo
        from myproject.utilities.GalaxyGroup import GalaxyGroup
        
        galaxyGroup, filter_func = args
        
        subhalos = list(galaxyGroup.getSubhalos())
        if len(subhalos) == 0:
            return None

        filtered_subhalos = [sh for sh in subhalos if filter_func(sh)]

        if len(filtered_subhalos) <= 1:  # need at least central + one satellite
            return None

        filtered_galaxyGroup = GalaxyGroup(
            galaxyGroup.getGroupID(),
            galaxyGroup.getRCrit200(),
            galaxyGroup.getMCrit200(),
            galaxyGroup.getPosCM(),
            galaxyGroup.getPos(),
            filtered_subhalos,
        )
        return filtered_galaxyGroup

    @staticmethod
    def _correct_positions_for_group(args : tuple[GalaxyGroup, float]):
        """Helper function to correct positions for a single galaxy group. Make all positions relative to central galaxy."""
        from myproject.utilities.Subhalo import Subhalo
        import numpy as np
        
        galaxyGroup, boxsize = args
        
        central_pos = galaxyGroup.getCentralSubhalo().getPosition()
        # Set central galaxy position to origin
        corrected_central_pos = np.zeros(3)
        
        galaxyGroupCM = galaxyGroup.getPosCM()
        # Correct group CM position
        corrected_galaxyGroupCM = ListGalaxyGroup.correctPositionWRTBoxsize(boxsize, central_pos, galaxyGroupCM)
        galaxyGroup.setPosCM(corrected_galaxyGroupCM)
        
        corrected_galaxyGroup_pos = ListGalaxyGroup.correctPositionWRTBoxsize(boxsize, central_pos, galaxyGroup.getPos())
        galaxyGroup.setPos(corrected_galaxyGroup_pos)
        
        # Correct all subhalo positions
        for subhalo in galaxyGroup.getSubhalos():
            position = subhalo.getPosition()
            corrected_position = np.zeros(3)
            for dim in range(3):
                rel_pos = position[dim] - central_pos[dim]
                # Wrap to [-boxsize/2, boxsize/2]
                if abs(rel_pos) > boxsize / 2:
                    if rel_pos > 0:
                        rel_pos -= boxsize
                    else:
                        rel_pos += boxsize
                # corrected_position[dim] = ((rel_pos + boxsize/2) % boxsize) - boxsize/2
                corrected_position[dim] = rel_pos
            subhalo.setPosition(corrected_position)
        
        galaxyGroup.getCentralSubhalo().setPosition(corrected_central_pos)
        
        return galaxyGroup
    
    @staticmethod
    def correctPositionWRTBoxsize(boxsize : float, posStatic : np.ndarray, relPos : np.ndarray):
        '''
        Corrects a position with respect to a static position considering periodic boundary conditions.
        
        :param boxsize: Size of the simulation box
        :param posStatic: Static reference position (e.g., central galaxy position)
        :param relPos: Relative position to be corrected
        :return: Corrected relative position
        '''
        corrected_position = np.zeros(3)
        for dim in range(3):
            delta = relPos[dim] - posStatic[dim]
            if abs(delta) > boxsize / 2:
                if delta > 0:
                    delta -= boxsize
                else:
                    delta += boxsize
            corrected_position[dim] = delta
        return corrected_position

    @staticmethod
    def _serialize_galaxy_group(args : tuple[GalaxyGroup, int]):
        """Helper function to serialize a galaxy group's data for HDF5 writing."""
        galaxyGroup, i = args
        
        group_data = {
            'index': i,
            'group_id': galaxyGroup.getGroupID(),
            'RCrit200': galaxyGroup.getRCrit200(),
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
                'luminosities': subhalo.getLuminosities(),
                'luminositiesSDSS': subhalo.getLuminositiesSDSS(),
                'joinTimes' : subhalo.getJoiningRedshiftInfo()
            }
            group_data['subhalos'].append(subhalo_data)
        
        return group_data
        