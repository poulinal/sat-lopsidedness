# ADP 2026

from myproject.utilities.Subhalo import Subhalo
from .GalaxyGroup import GalaxyGroup
import h5py as h5
import numpy as np

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
        
    def getNumGalaxyGroups(self):
        return self.lenGalaxyGroups
    
    def getGalaxyGroupI(self, i):
        return self.listGalaxyGroups[i]
    
    def getListPairwiseDifferences(self) -> list[list[tuple[float, float, float]]]:
        return self.list_pairwise_differences
            
    def compute_all_pairwise_polar_differences(self) -> list[list[tuple[float, float, float]]]:
        '''
        Docstring for compute_all_pairwise_polar_differences
        Computes all pairwise polar angle differences (in each plane: XY, YZ, ZX) between satellite galaxies in each galaxy group with resepect to the host galaxy.
        Returns a list of all pairwise polar angle differences. Indicies go from 0 to 180º. Indicie of 0 means aligned, 180º means anti-aligned. Values correspond to the density/probability of finding satellite galaxies at a given polar angle difference.
        
        Note: Each entry in the returned list corresponds to a galaxy group, containing a list of tuples. Each tuple contains the polar angle differences (in degrees) between a pair of satellite galaxies in the XY, YZ, and ZX planes respectively. NOT the average of the three planes.
        
        :param self: Description
        :return: Returns a list where each indicie corresponds to a galaxy group, each containing a list of pairwise polar angle differences between satellite galaxies.
        :rtype: list[list[float]]
        '''
        for galaxyGroup in self.listGalaxyGroups:
            print(f"Progress: Processing Galaxy Group ID {galaxyGroup.getGroupID()} / {len(self.listGalaxyGroups)}", end='\r')
            subhalos : list[Subhalo] = galaxyGroup.getSubhalos()
            num_subhalos = len(subhalos)
            central_pos = galaxyGroup.getPos()  # XYZ
            group_pairwise_differences = []
            for i in range(num_subhalos):
                for j in range(i + 1, num_subhalos):
                    pos_i = subhalos[i].getPosition() # XYZ
                    pos_j = subhalos[j].getPosition() # XYZ
                    # Compute polar angle difference in XY plane
                    vec_i_xy = np.array([pos_i[0] - central_pos[0], pos_i[1] - central_pos[1]])
                    vec_j_xy = np.array([pos_j[0] - central_pos[0], pos_j[1] - central_pos[1]])
                    angle_i_xy = np.arctan2(vec_i_xy[1], vec_i_xy[0])
                    angle_j_xy = np.arctan2(vec_j_xy[1], vec_j_xy[0])
                    diff_xy = np.abs(angle_i_xy - angle_j_xy) * (180.0 / np.pi)
                    diff_xy = diff_xy if diff_xy <= 180 else 360 - diff_xy
                    
                    # Compute polar angle difference in YZ plane
                    vec_i_yz = np.array([pos_i[1] - central_pos[1], pos_i[2] - central_pos[2]])
                    vec_j_yz = np.array([pos_j[1] - central_pos[1], pos_j[2] - central_pos[2]])
                    angle_i_yz = np.arctan2(vec_i_yz[1], vec_i_yz[0])
                    angle_j_yz = np.arctan2(vec_j_yz[1], vec_j_yz[0])
                    diff_yz = np.abs(angle_i_yz - angle_j_yz) * (180.0 / np.pi)
                    diff_yz = diff_yz if diff_yz <= 180 else 360 - diff_yz

                    # Compute polar angle difference in ZX plane
                    vec_i_zx = np.array([pos_i[2] - central_pos[2], pos_i[0] - central_pos[0]])
                    vec_j_zx = np.array([pos_j[2] - central_pos[2], pos_j[0] - central_pos[0]])
                    angle_i_zx = np.arctan2(vec_i_zx[1], vec_i_zx[0])
                    angle_j_zx = np.arctan2(vec_j_zx[1], vec_j_zx[0])
                    diff_zx = np.abs(angle_i_zx - angle_j_zx) * (180.0 / np.pi)
                    diff_zx = diff_zx if diff_zx <= 180 else 360 - diff_zx
                    
                    pairwise_difference = (diff_xy, diff_yz, diff_zx)
                    group_pairwise_differences.append(pairwise_difference)

            self.list_pairwise_differences.append(group_pairwise_differences)
            
        return self.list_pairwise_differences
    
    def compute_probablity_distribution_of_polar_differences(self, bin_size : float=5.0) -> tuple[np.ndarray, np.ndarray]:
        '''
        Docstring for compute_probablity_distribution_of_polar_differences
        Computes the probability distribution of polar angle differences between satellite galaxies in each galaxy group with resepect to the host galaxy.
        Returns a tuple containing the bin centers and the corresponding probability densities.
        
        :param self: Description
        :param bin_size: Size of the bins for the histogram (default is 5.0 degrees)
        :return: Tuple of (bin_centers, probability_densities)
        :rtype: tuple[np.ndarray, np.ndarray]
        '''
        pairwise_differences_flatten = []
        for galaxyPairwiseGroup in self.list_pairwise_differences:
            for pair in galaxyPairwiseGroup:
                pairwise_differences_flatten.extend(pair)  # Unpack the tuple and add each angle difference
        bins = np.arange(0, 180 + bin_size, bin_size)
        hist, bin_edges = np.histogram(pairwise_differences_flatten, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, hist
    
    def compute_all_MRL_directionality(self) -> list[float]:
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
        if not self.list_pairwise_differences:
            self.compute_all_pairwise_polar_differences()
        else:
            print("Using pre-computed pairwise polar differences.")
            
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
        
    def filterSubhalos(self, minStellarMass : float=None, maxStellarMass : float=None, minHalfMassRad_kpc : float=None, maxHalfMassRad_kpc : float=None, centralPosTolerance_kpc : float=1000) -> None:
        '''
        Modifies list_galaxy_groups and Filters subhalos in each galaxy group based on specified criteria.
        - remove non cosmlogoical in origin (subhaloflag = 0)
        - only mass greater than 13 solar mass
        - remove groups where no single central (i.e. central pos does not match cm pos)
        - Later on will need to filter based on radius due to smoothing length: because the smoothing length used by the TNG300 is different for z < 1 and z >= 1 we'll likely want to adopt a minimum  size for the galaxies we want to use following Curtis et al. (2026) (note: we need to check to see if TNGCluster used the same smoothing length change as TNG300)
        
        :param minStellarMass: Minimum stellar mass to retain a subhalo (default is None)
        :param maxStellarMass: Maximum stellar mass to retain a subhalo (default is None)
        :param minHalfMassRad_kpc: Minimum half-mass radius in kpc to retain a subhalo (default is None)
        :param maxHalfMassRad_kpc: Maximum half-mass radius in kpc to retain a subhalo (default is None)
        :param centralPosTolerance_kpc: Maximum distance from central position in kpc to retain a subhalo (default is 1000 kpc, i.e. 1 Mpc)
        '''
        list_filtered_galaxy_groups : list[GalaxyGroup]= []
        for galaxyGroup in self.listGalaxyGroups:
            # print(f"Progress: Filtering Galaxy Group ID {galaxyGroup.getGroupID()} / {len(self.listGalaxyGroups)}", end='\r')
            filtered_subhalos : list[Subhalo] = []
            central_pos = galaxyGroup.getPos()
            galaxyGroupCM = galaxyGroup.getPosCM()
            
            distance = np.linalg.norm(central_pos - galaxyGroupCM)
            if distance > centralPosTolerance_kpc:
                print(f"Skipping Galaxy Group ID {galaxyGroup.getGroupID()} due to central position mismatch (distance: {distance} kpc)")
                continue
            
            for i, subhalo in enumerate(galaxyGroup.getSubhalos()):
                print(f"Progress: Filtering Subhalo {i} / {galaxyGroup.getNumSubhalos()} of galaxy:  {galaxyGroup.getGroupID()} / {len(self.listGalaxyGroups)}", end='\r')
                if subhalo.getFlag() == 0:
                    continue
                if minStellarMass is not None and subhalo.getStellarMass() < minStellarMass:
                    continue
                if maxStellarMass is not None and subhalo.getStellarMass() > maxStellarMass:
                    continue
                if minHalfMassRad_kpc is not None and subhalo.getHalfMassRad() < minHalfMassRad_kpc:
                    continue
                if maxHalfMassRad_kpc is not None and subhalo.getHalfMassRad() > maxHalfMassRad_kpc:
                    continue
                filtered_subhalos.append(subhalo)
                
            filtered_galaxyGroup = GalaxyGroup(galaxyGroup.getGroupID(), galaxyGroup.getMCrit200(), galaxyGroup.getPosCM(), galaxyGroup.getPos(), filtered_subhalos)
            if len(filtered_subhalos) > 0:
                list_filtered_galaxy_groups.append(filtered_galaxyGroup)
                
        self.listGalaxyGroups = list_filtered_galaxy_groups
                        
    def correctPositions(self, boxsize : float):
        '''
        Corrects the positions of subhalos in each galaxy group to account for periodic boundary conditions. 
        Ensure all positions are relative to the central galaxy.
        
        :param boxsize: Size of the simulation box
        '''
        for galaxyGroup in self.listGalaxyGroups:
            # print(f"Progress: Correcting Positions for Galaxy Group ID {galaxyGroup.getGroupID()} / {len(self.listGalaxyGroups)}", end='\r')
            central_pos = galaxyGroup.getPos()
            galaxyGroupCM = galaxyGroup.getPosCM()
            
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
            
            corrected_central_pos = np.zeros(3)
            galaxyGroup.setPos(corrected_central_pos)
            
            
            
            for subhalo in galaxyGroup.getSubhalos():
                print(f"Progress: Correcting Subhalo {i} / {galaxyGroup.getNumSubhalos()} of galaxy:  {galaxyGroup.getGroupID()} / {len(self.listGalaxyGroups)}", end='\r')
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
        
        
    
    def save_to_hdf5(self, h5file : h5.File, overwrite : bool=True):
        
        if not overwrite:
            #check which groups already exist
            existing_groups = set(h5file['GalaxyGroups'].keys()) if 'GalaxyGroups' in h5file else set()
            for i, galaxyGroup in enumerate(self.listGalaxyGroups):
                if f'GalaxyGroup_{i}' in existing_groups:
                    print(f"GalaxyGroup_{i} already exists in file. Skipping.")
                    continue
        
        else:
            for key, value in self.headerInformation.items():
                h5file.attrs[key] = value
            
            grp = h5file.create_group('GalaxyGroups')
            for i, galaxyGroup in enumerate(self.listGalaxyGroups):
                print(f"Progress: {i+1}/{len(self.listGalaxyGroups)}", end='\r')
                gg_grp : h5.Group = grp.create_group(f'GalaxyGroup_{i}')
                gg_grp.attrs['group_id'] = galaxyGroup.getGroupID()
                gg_grp.attrs['MCrit200'] = galaxyGroup.getMCrit200()
                gg_grp.attrs['posCM'] = galaxyGroup.getPosCM()
                gg_grp.attrs['pos'] = galaxyGroup.getPos()
                
                subhalos_grp : h5.Group = gg_grp.create_group('Subhalos')
                for j, subhalo in enumerate(galaxyGroup.getSubhalos()):
                    subhalo : Subhalo = subhalo
                    sh_grp = subhalos_grp.create_group(f'Subhalo_{j}')
                    sh_grp.attrs['idx'] = subhalo.getIdx()
                    sh_grp.attrs['flag'] = subhalo.getFlag()
                    sh_grp.attrs['mass'] = subhalo.getMass()
                    sh_grp.attrs['stellarMass'] = subhalo.getStellarMass()
                    sh_grp.attrs['groupNumber'] = subhalo.getGroupNumber()
                    sh_grp.attrs['position'] = subhalo.getPosition()
                    sh_grp.attrs['halfMassRad'] = subhalo.getHalfMassRad()
                    sh_grp.attrs['vmaxRadius'] = subhalo.getVmaxRadius()
                    sh_grp.attrs['luminosities'] = subhalo.getLuminosities()
                    
    def load_from_hdf5(self, h5file : h5.File):
        self.listGalaxyGroups = []
        self.headerInformation = {}
        for key, value in h5file.attrs.items():
            self.headerInformation[key] = value
        grp = h5file['GalaxyGroups']
        for i, gg_key in enumerate(grp):
            print(f"Progress: {i+1}/{len(grp)}", end='\r')
            gg_grp : h5.Group = grp[gg_key]
            group_id = gg_grp.attrs['group_id']
            MCrit200 = gg_grp.attrs['MCrit200']
            posCM = gg_grp.attrs['posCM']
            pos = gg_grp.attrs['pos']
            galaxyGroup = GalaxyGroup(group_id, MCrit200, posCM, pos)
            
            subhalos_grp : h5.Group = gg_grp['Subhalos']
            for sh_key in subhalos_grp:
                sh_grp : h5.Group = subhalos_grp[sh_key]
                idx = sh_grp.attrs['idx']
                flag = sh_grp.attrs['flag']
                mass = sh_grp.attrs['mass']
                stellarMass = sh_grp.attrs['stellarMass']
                groupNumber = sh_grp.attrs['groupNumber']
                position = sh_grp.attrs['position']
                halfMassRad = sh_grp.attrs['halfMassRad']
                vmaxRadius = sh_grp.attrs['vmaxRadius']
                luminosities = sh_grp.attrs['luminosities']
                
                subhalo = Subhalo(idx, flag, mass, stellarMass, groupNumber, position, halfMassRad, vmaxRadius, luminosities)
                galaxyGroup.addSubhalo(subhalo)
            
            self.listGalaxyGroups.append(galaxyGroup)
            
            
            
        