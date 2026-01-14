# ADP 2026

from myproject.utilities.Subhalo import Subhalo
from .GalaxyGroup import GalaxyGroup
import h5py as h5
import numpy as np

class ListGalaxyGroup:
    def __init__(self, listGalaxyGroups : list[GalaxyGroup]=[], headerInformation : dict={}):
        self.listGalaxyGroups = listGalaxyGroups
        self.headerInformation = headerInformation
        
    @classmethod
    def from_hdf5(cls, h5file : h5.File):
        instance = cls()
        instance.load_from_hdf5(h5file)
        return instance
        
    def addGalaxyGroup(self, galaxyGroup : GalaxyGroup):
        self.listGalaxyGroups.append(galaxyGroup)
        
    def getNumGalaxyGroups(self):
        return len(self.listGalaxyGroups)
    
    def save_to_hdf5(self, h5file : h5.File):
        
        for key, value in self.headerInformation.items():
            h5file.attrs[key] = value
        
        grp = h5file.create_group('GalaxyGroups')
        for i, galaxyGroup in enumerate(self.listGalaxyGroups):
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
        for gg_key in grp:
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
            
            
            
    def compute_all_pairwise_polar_differences(self) -> list[float]:
        '''
        Docstring for compute_all_pairwise_polar_differences
        Computes all pairwise polar angle differences between satellite galaxies in each galaxy group.
        Returns a list of all pairwise polar angle differences. Indicies go from 0 to 180º. Indicie of 0 means aligned, 180º means anti-aligned. Values correspond to the density/probability of finding satellite galaxies at a given polar angle difference.
        
        :param self: Description
        :return: Description
        :rtype: list[float]
        '''
        pairwise_differences = []
        for galaxyGroup in self.listGalaxyGroups:
            subhalos = galaxyGroup.getSubhalos()
            num_subhalos = len(subhalos)
            for i in range(num_subhalos):
                for j in range(i + 1, num_subhalos):
                    pos_i = subhalos[i].getPosition()
                    pos_j = subhalos[j].getPosition()
                    # Compute polar angle difference
                    angle_i = np.arctan2(pos_i[1], pos_i[0])
                    angle_j = np.arctan2(pos_j[1], pos_j[0])
                    diff = np.abs(angle_i - angle_j)
                    pairwise_differences.append(diff)
        return pairwise_differences
    
    def compute_all_MRL_directionality(self) -> list[float]:
        '''
        Docstring for compute_MRL_directionality
        Computes the Mean Resultant Length (MRL) directionality for each galaxy group.
        Returns a list of MRL directionality values for each galaxy group. Indicie goes from 0 to 1, where 0 indicates isotropic distribution and 1 indicates satellites clustering toward one direction. Values correspond to the density/probability of finding satellite galaxies aligned in a in the MRL.
        
        :param self: Description
        :return: Description
        :rtype: list[float]
        '''
        MRL_values = []
        for galaxyGroup in self.listGalaxyGroups:
            subhalos = galaxyGroup.getSubhalos()
            # Placeholder for MRL calculation
            MRL_value = len(subhalos)  # Example: using number of subhalos as a placeholder
            MRL_values.append(MRL_value)
        return MRL_values
            
            