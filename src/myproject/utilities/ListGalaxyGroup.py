# ADP 2026

from myproject.utilities.Subhalo import Subhalo
from .GalaxyGroup import GalaxyGroup
from .parallelTools import parallel_map, get_optimal_processes
import h5py as h5
import numpy as np
import os
import pickle
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

    def getHeaderInformation(self):
        return self.headerInformation
        
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
    
    def getGalaxyGroupI(self, i) -> GalaxyGroup:
        return self.listGalaxyGroups[i]

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
        return (min(num_subhalos_list), max(num_subhalos_list))
    
    def getListPairwiseDifferences(self) -> list[list[tuple[float, float, float]]]:
        return self.list_pairwise_differences
            
    def compute_all_pairwise_polar_differences(self, parallelize : bool=False, n_processes : Optional[int]=None, tempSaveDir : str=None, rewrite: bool=False) -> list[list[tuple[float, float, float]]]:
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
        print(f"Total Galaxy Groups to process: {len(self.listGalaxyGroups)} with pairs : {sum([gg.getNumSubhalos() * (gg.getNumSubhalos() - 1) // 2 for gg in self.listGalaxyGroups])}")
        

        
        if parallelize:
            if n_processes is None:
                n_processes = get_optimal_processes(len(self.listGalaxyGroups))
        
            print(f"Computing pairwise differences in parallel with {n_processes} processes...")
            total = len(self.listGalaxyGroups)
            
            import multiprocessing as mp
            with mp.Pool(processes=n_processes) as pool:
                for i, result in enumerate(pool.imap(ListGalaxyGroup._compute_pairwise_for_group, self.listGalaxyGroups), 1):
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
                if existing_files:
                    #get the index of each file
                    existing_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
                    last_file = existing_files[-1]
                    print(f"Resuming from existing temp file: {last_file}")
                    start_index = int(last_file.split('_')[2].split('.')[0])
                
            for i, galaxyGroup in enumerate(self.listGalaxyGroups, 1):
                if rewrite == False and tempSaveDir is not None and i <= start_index:
                    continue  # Skip already processed groups
                print(f"Progress: Processing Galaxy Group ID {galaxyGroup.getGroupID()} / {len(self.listGalaxyGroups)}", end='\r', flush=True)
                group_pairwise_differences = []
                group_pairwise_differences = ListGalaxyGroup._compute_pairwise_for_group(galaxyGroup)

                if tempSaveDir is not None:
                    batch_list_pairwise_differences.append(group_pairwise_differences)
                else:
                    self.list_pairwise_differences.append(group_pairwise_differences)
                
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
            
        return self.list_pairwise_differences
    
    def compute_probablity_distribution_of_polar_differences(self, bins : np.ndarray =np.arange(0, 180 + 5, 5), parallelize: bool = False, tempSaveDir : str=None, rewrite: bool=False) -> tuple[np.ndarray, np.ndarray]:
        '''
        Docstring for compute_probablity_distribution_of_polar_differences
        Computes the probability distribution of polar angle differences between satellite galaxies in each galaxy group with resepect to the host galaxy.
        Returns a tuple containing the bin centers and the corresponding probability densities.
        
        :param self: Description
        :param bins: Array of bin edges for the histogram (default is np.arange(0, 180 + 5, 5))
        :return: Tuple of (bin_centers, probability_densities)
        :rtype: tuple[np.ndarray, np.ndarray]
        '''
        if not self.list_pairwise_differences:
            self.compute_all_pairwise_polar_differences(parallelize=parallelize, n_processes=None, tempSaveDir=tempSaveDir, rewrite=rewrite)
        pairwise_differences_flatten = []
        # for galaxyPairwiseGroup in self.list_pairwise_differences:
        #     for pair in galaxyPairwiseGroup:
        #         pairwise_differences_flatten.extend(pair)  # Unpack the tuple and add each angle difference
        
         #since already flattened in temp save
        for galaxyPairwiseGroup in self.list_pairwise_differences:
            pairwise_differences_flatten.extend(galaxyPairwiseGroup)
        # bins = np.arange(0, 180 + bin_size, bin_size)
        print(f"bins: {bins}")
        hist, bin_edges = np.histogram(pairwise_differences_flatten, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        Numbers_in_bins, _ = np.histogram(pairwise_differences_flatten, bins=bins)
        errorbars = np.sqrt(Numbers_in_bins) / np.sum(Numbers_in_bins) / (bin_edges[1] - bin_edges[0])  # Poisson errors normalized to density
        return bin_centers, hist, errorbars
    
    def compute_all_MRL_directionality(self, parallelize: bool = False, n_processes: Optional[int] = None, tempSaveDir : str=None, rewrite: bool=False) -> list[float]:
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
                if existing_files:
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
            
    def compute_probablity_distribution_of_MRL_directionality(self, bin_size : float=0.05, parallelize: bool=False, n_processes: Optional[int]=None, tempSaveDir: Optional[str]=None, rewrite: bool=False) -> tuple[np.ndarray, np.ndarray]:
        '''
        Docstring for compute_probablity_distribution_of_MRL_directionality
        Computes the probability distribution of Mean Resultant Length (MRL) directionality for each galaxy group.
        Returns a tuple containing the bin centers, the corresponding probability densities, and errorbars based on (for N values in a bin, the error for that bin is sqrt{N}).
        
        :param self: Description
        :param bin_size: Size of the bins for the histogram (default is 0.05)
        :return: Tuple of (bin_centers, probability_densities, errorbars)
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        '''
        if not self.MRL_values:
            self.compute_all_MRL_directionality(parallelize=parallelize, n_processes=n_processes, tempSaveDir=tempSaveDir, rewrite=rewrite)
        else:
            print("Using pre-computed MRL directionality values.")
        bins = np.arange(0, 1 + bin_size, bin_size)
        hist, bin_edges = np.histogram(self.MRL_values, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        Numbers_in_bins, _ = np.histogram(self.MRL_values, bins=bins)
        errorbars = np.sqrt(Numbers_in_bins) / np.sum(Numbers_in_bins) / (bin_edges[1] - bin_edges[0])  # Poisson errors normalized to density
        # Print how many values are in each bin
        for i, count in enumerate(Numbers_in_bins):
            print(f"Bin {i} ({bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}): {count} values")
        return bin_centers, hist, errorbars
        
    def filterSubhalos(self, minGGMass : float=None, maxGGMass : float=None, minSatStellarMass : float=None, maxSatStellarMass : float=None, minHalfMassRad_kpc : float=None, maxHalfMassRad_kpc : float=None, centralPosTolerance_kpc : Optional[float]=None, M_r_min : float=None, M_r_max : float=None, satWithinR200 : bool = False, redGalaxies : bool = False, blueGalaxies : bool = False, minNumGalaxies : Optional[int]=None, maxNumGalaxies : Optional[int]=None, withinXPercentR200 : tuple[float, float] = None, parallelize : bool=False, n_processes: Optional[int]=None) -> None:
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
        :param minNumGalaxies: Minimum number of satellite galaxies required in a galaxy group to retain it (default is None)
        :param maxNumGalaxies: Maximum number of satellite galaxies allowed in a galaxy group to retain it (default is None)
        :param withinXPercentR200: Tuple specifying the range (min, max) as a fraction of R200 within which to retain satellites (default is None)
        :param parallelize: Whether to parallelize the filtering process (default is False)
        :param n_processes: Number of processes to use if parallelizing (default is None, which uses optimal number)
        '''
        list_filtered_galaxy_groups : list[GalaxyGroup]= []
        # total = len(self.listGalaxyGroups)
            
        # Prepare arguments for parallel processing
        args_list = [
            (gg, minGGMass, maxGGMass, minSatStellarMass, maxSatStellarMass, minHalfMassRad_kpc, maxHalfMassRad_kpc, centralPosTolerance_kpc, M_r_min, M_r_max, satWithinR200, redGalaxies, blueGalaxies, minNumGalaxies, maxNumGalaxies, withinXPercentR200)
            for gg in self.listGalaxyGroups
        ]
        total = len(args_list)
        print(f"total to process: {total}")
        
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
        return list_filtered_galaxy_groups
                        

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
            
        else:
            results = []
            for args in args_list:
                print(f"Progress: Processing Galaxy Group ID {args[0].getGroupID()} / {len(self.listGalaxyGroups)}", end='\r')
                results.append(ListGalaxyGroup._correct_positions_for_group(args))
                
        # self.setGalaxyGroups(results)
        # return self.getAllGalaxyGroups()
        return results
       
       
       
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
                gg_grp.attrs['RCrit200'] = group_data['RCrit200']
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
                for i, result in enumerate(pool.imap(ListGalaxyGroup._load_group_from_hdf5, args_list), 1):
                    results.append(result)
                    percent = (i / total) * 100
                    print(f"\rProgress: {i}/{total} ({percent:.1f}%)", end='', flush=True)
                print()  # New line after progress
            
            self.listGalaxyGroups = results
        else:
            for i, gg_key in enumerate(gg_keys):
                print(f"Progress: {i+1}/{total}", end='\r')
                galaxyGroup = ListGalaxyGroup._load_group_from_hdf5((h5file.filename, gg_key))
                self.listGalaxyGroups.append(galaxyGroup)
        
        self.lenGalaxyGroups = len(self.listGalaxyGroups)
        print(f"\nLoaded {len(self.listGalaxyGroups)} galaxy groups from HDF5.")

    # Standalone functions for multiprocessing (must be picklable)
    @staticmethod
    def _load_group_from_hdf5(args):
        """Helper function to load a single galaxy group from HDF5 file."""
        import h5py as h5
        from myproject.utilities.Subhalo import Subhalo
        
        h5_filename, gg_key = args
        
        with h5.File(h5_filename, 'r') as h5file:
            gg_grp = h5file['GalaxyGroups'][gg_key]
            galaxy_group_id = gg_grp.attrs['group_id']
            RCrit200 = gg_grp.attrs['RCrit200']
            MCrit200 = gg_grp.attrs['MCrit200']
            posCM = gg_grp.attrs['posCM']
            pos = gg_grp.attrs['pos']
            galaxyGroup = GalaxyGroup(galaxy_group_id, RCrit200, MCrit200, posCM, pos, listSubhalos=[])
            
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
                
                subhalo = Subhalo(idx, group_id, flag, mass, stellarMass, groupNumber, position, halfMassRad, vmaxRadius, luminosities)
                galaxyGroup.addSubhalo(subhalo)
        
        return galaxyGroup

    @staticmethod
    def _compute_pairwise_for_group(galaxyGroup : GalaxyGroup) -> list[tuple[float, float, float]]:
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

        angles_xy = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])
        angles_yz = np.arctan2(rel_pos[:, 2], rel_pos[:, 1])
        angles_xz = np.arctan2(rel_pos[:, 0], rel_pos[:, 2])
        

        # Stream-compute pairwise differences without allocating full matrices
        group_pairwise_differences = []
        deg = 180.0 / np.pi
        
        for i in range(num_subhalos):
            print(f"    Processing Subhalo {i+1}/{num_subhalos} in Galaxy Group ID {galaxyGroup.getGroupID()} with total pairs {num_subhalos * (num_subhalos - 1) // 2}")
            for j in range(i + 1, num_subhalos):
                # Compute angle differences for each plane (in radians, then convert)
                # where delta_xy=0 corresponds to the same side and delta_xy=180 corresponds to opposite sides
                delta_xy = angles_xy[i] - angles_xy[j]
                delta_yz = angles_yz[i] - angles_yz[j]
                delta_xz = angles_xz[i] - angles_xz[j]
                
                # Normalize differences to [0, π]
                diff_xy = np.abs((delta_xy + np.pi) % (2 * np.pi) - np.pi)
                diff_yz = np.abs((delta_yz + np.pi) % (2 * np.pi) - np.pi)
                diff_xz = np.abs((delta_xz + np.pi) % (2 * np.pi) - np.pi)
                
                # Convert to degrees
                pairwise_difference = (diff_xy * deg, diff_yz * deg, diff_xz * deg)
                group_pairwise_differences.append(pairwise_difference)
        
        return group_pairwise_differences
    
    @staticmethod
    def _compute_MRL_for_group(galaxyGroup : GalaxyGroup) -> list[float]:
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
        angles_xy = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])  # XY plane: arctan2(y, x)
        angles_yz = np.arctan2(rel_pos[:, 2], rel_pos[:, 1])  # YZ plane: arctan2(z, y)
        angles_xz = np.arctan2(rel_pos[:, 2], rel_pos[:, 0])  # XZ plane: arctan2(x, z)
        
        # #compute angle based on cosine to avoid issues with arctan2
        # # r = np.sqrt(rel_pos[:, 0]**2 + rel_pos[:, 1]**2)
        # r_xy = np.linalg.norm(rel_pos[:, :2], axis=1)  # sqrt(x^2 + y^2) per row
        # r_yz = np.linalg.norm(rel_pos[:, 1:], axis=1)  # sqrt(y^2 + z^2) per row
        # r_xz = np.linalg.norm(rel_pos[:, [2, 0]], axis=1)  # sqrt(z^2 + x^2) per row

        # print(f"r_xy: {r_xy}") if id == 0 else None
        # print(f"r_yz: {r_yz}") if id == 0 else None
        # print(f"r_xz: {r_xz}") if id == 0 else None
        # # r = np.linalg.norm(rel_pos, axis=1)  # sqrt(x^2 + y^2 + z^2) per row
        # cos_theta_xy = abs(rel_pos[:, 0]) / r_xy # cos(theta) = x/r
        # cos_theta_yz = abs(rel_pos[:, 1]) / r_yz # cos(theta) = y/r
        # cos_theta_xz = abs(rel_pos[:, 2]) / r_xz # cos(theta) = z/r
        
        # angles_xy = np.arccos(cos_theta_xy)
        # angles_yz = np.arccos(cos_theta_yz)
        # angles_xz = np.arccos(cos_theta_xz)
        
        # #normalize to [0, 2pi]
        # angles_xy = angles_xy % (2 * np.pi)
        # angles_yz = angles_yz % (2 * np.pi)
        # angles_xz = angles_xz % (2 * np.pi)
        
        # Compute MRL for XY plane
        cos_sum_xy = np.sum(np.cos(angles_xy))
        sin_sum_xy = np.sum(np.sin(angles_xy))
        R_xy = (1/n) * np.sqrt(cos_sum_xy**2 + sin_sum_xy**2)
        
        # Compute MRL for YZ plane
        cos_sum_yz = np.sum(np.cos(angles_yz))
        sin_sum_yz = np.sum(np.sin(angles_yz))
        R_yz = (1/n) * np.sqrt(cos_sum_yz**2 + sin_sum_yz**2)
        
        # Compute MRL for XZ plane
        cos_sum_xz = np.sum(np.cos(angles_xz))
        sin_sum_xz = np.sum(np.sin(angles_xz))
        R_xz = (1/n) * np.sqrt(cos_sum_xz**2 + sin_sum_xz**2)

        print(f"final: {[R_xy, R_yz, R_xz]}") if id == 0 else None
        
        return [R_xy, R_yz, R_xz]

    @staticmethod
    def _filter_subhalos_for_group(args : tuple[GalaxyGroup, float, float, float, float, float, float, float, float, float, bool, bool, bool, int, int, tuple[float, float]]):
        """Helper function to filter subhalos for a single galaxy group."""
        from myproject.utilities.Subhalo import Subhalo
        from myproject.utilities.GalaxyGroup import GalaxyGroup
        import numpy as np
        
        galaxyGroup, minGGMass, maxGGMass, minSatStellarMass, maxSatStellarMass, minHalfMassRad_kpc, maxHalfMassRad_kpc, centralPosTolerance_kpc, M_r_min, M_r_max, satWithinR200, redGalaxies, blueGalaxies, minNumGalaxies, maxNumGalaxies, withinXPercentR200 = args
        
        # print(f"masses: {minGGMass, galaxyGroup.getMCrit200()}")
        if minGGMass is not None and galaxyGroup.getMCrit200() <= minGGMass:
            return None
        if maxGGMass is not None and galaxyGroup.getMCrit200() >= maxGGMass:
            return None
        
        central_pos = galaxyGroup.getCentralSubhalo().getPosition()
        galaxyGroupCM = galaxyGroup.getPosCM()
        
        distance = np.linalg.norm(central_pos - galaxyGroupCM)
        if centralPosTolerance_kpc is not None and distance > centralPosTolerance_kpc:
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
            if M_r_min is not None and (np.isnan(subhalo.getRbandMagnitude()) or subhalo.getRbandMagnitude() <= M_r_min):
                continue
            if M_r_max is not None and (np.isnan(subhalo.getRbandMagnitude()) or subhalo.getRbandMagnitude() >= M_r_max):
                continue
            if satWithinR200 and subhalo != galaxyGroup.getCentralSubhalo():
                distance_to_central = np.linalg.norm(subhalo.getPosition() - central_pos)
                # print(f"distance: {distance_to_central} to {galaxyGroup.getRCrit200()}")
                if distance_to_central > galaxyGroup.getRCrit200():
                    continue
                
            if (redGalaxies or blueGalaxies) and subhalo != galaxyGroup.getCentralSubhalo():
                g_mag = subhalo.getGbandMagnitude()
                r_mag = subhalo.getRbandMagnitude()
                if np.isnan(g_mag) or np.isnan(r_mag):
                    print("WARNING... np.nan")
                    continue  # Skip if magnitudes are not available
                g_r_color = g_mag - r_mag
                if redGalaxies and g_r_color < 0.65:
                    continue
                if blueGalaxies and g_r_color >= 0.65:
                    continue
                
            if withinXPercentR200 is not None and subhalo != galaxyGroup.getCentralSubhalo():
                distance_to_central = np.linalg.norm(subhalo.getPosition() - central_pos)
                r200 = galaxyGroup.getRCrit200()
                min_radius = withinXPercentR200[0] * r200
                max_radius = withinXPercentR200[1] * r200
                if distance_to_central < min_radius or distance_to_central > max_radius:
                    continue
            # If we made it past all filters, add the subhalo
            filtered_subhalos.append(subhalo)
        
        if len(filtered_subhalos) <= 1: #since central
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
            filtered_subhalos
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
                'luminosities': subhalo.getLuminosities()
            }
            group_data['subhalos'].append(subhalo_data)
        
        return group_data
        