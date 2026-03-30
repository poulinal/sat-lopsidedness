#AP 2026
from myproject.utilities import iapi_TNG, Subhalo, GalaxyGroup, ListGalaxyGroup
from myproject.utilities.joinTime import JoinTime
import os
import h5py as h5
import numpy as np
import multiprocessing as mp
from myproject.utilities.parallelTools import get_optimal_processes

from typing import Optional

class GalaxyGroupData:
    def __init__(self, sim : str = 'TNG300-1', snapshot : int = 99):
        self.sim = sim
        self.snapshot = snapshot
        # sim = 'TNG300-1'
        # snapshot = 99
        self.snapshot_dic = {99: (0, 'z0p0'), 91: (0.1, 'z0p1'), 84: (0.2, 'z0p2'), 78: (0.3, 'z0p3'), 72: (0.4, 'z0p4'), 67: (0.5, 'z0p5'), 59: (0.7, 'z0p7'), 50: (1.0, 'z1p0'), 40: (1.5, 'z1p5'), 33: (2.0, 'z2p0'), 25: (3.0, 'z3p0')}
        possible_snapshots = list(self.snapshot_dic.keys())
        self.TNG300_1_volsize = 302.6 #Mpc
        self.TNG_Cluster_volsize = 1003.8
        self.TNG_Cluster_boxsize=680
        self.TNG300_1_boxsize = 205 #Mpc/h
        self.scratchDataDirc = f'/scratch/poulin.al/lopsided/{self.sim}/{self.snapshot_dic[self.snapshot][1]}/data'
        self.scratchPlotDirc = f'/scratch/poulin.al/lopsided/{self.sim}/{self.snapshot_dic[self.snapshot][1]}/plots'
        
        # print(os.getenv("API_KEY"))



        baseUrl = 'http://www.tng-project.org/api/'
        r=iapi_TNG.get(baseUrl)
        print("r: ", r)
        #check the properties of the simulation you have selected
        self.simUrl = baseUrl+self.sim
        print(self.simUrl) 
        self.simdata : dict[str, object] = iapi_TNG.get(self.simUrl)
        print(self.simdata['description'])
        self.simdata.keys()




        self.h = self.simdata.get('hubble')
        print(f"hubble: {self.h}")

        #convert boxsize to kpc from Mpc/h
        self.TNG300_1_boxsize_kpc = self.TNG300_1_boxsize * 1e3 /self.h #kpc
        self.TNG_Cluster_boxsize_kpc = self.TNG_Cluster_boxsize * 1e3 / self.h
        self.sim_boxsize_kpc = self.TNG_Cluster_boxsize_kpc if self.sim == 'TNG-Cluster' else self.TNG300_1_boxsize_kpc


        self.z = self.snapshot_dic[self.snapshot][0] #redshift
        self.a = 1.0 / (1 + self.z)  # scale factor, where z = redshift
        
        self.initializeDataStructures()
        
        # self.getSubhaloData(sim, snapshot, subhalo_id)
        # self.getGroupData(sim, snapshot, subhalo_id)
        
    def initializeDataStructures(self):
        self.flag = None
        self.mass = None
        self.stellar_mass = None
        self.subhaloGroupNum = None
        self.subhaloPos = None
        self.subhaloHalfmassRad = None
        self.SubhaloVmaxRad = None
        self.SubhaloStellarPhotometrics = None
        
        self.groupMCrit200 = None
        self.groupRCrit200 = None
        self.groupCM = None
        self.groupPos = None
        self.maxValidGroupIndex = None
        
        self.list_of_galaxy_groups = None
        self.filtered_and_corrected_list_of_galaxy_groups = None
        
    def computeAllData(self, additionalFileIdentifier:str=''):
        for sim in [self.sim]:
            for i, snapshot in enumerate(self.snapshot_dic.keys()): #possible_snapshots:
                if snapshot != 99:
                    print(f"previously got all data for {snapshot}")
                    continue
                # if snapshot > 60:
                #     print(f"previously got all data for {snapshot}")
                #     continue
                if snapshot < 50:
                    print(f"Skipping redshifts less than snapshot 50 aka redshift 1.5 due to smoothing length change, {snapshot}")
                    continue
                subhalo_id = None #placeholder since we will be fetching all subhalos
                self.__init__(sim, snapshot)
                print(f"New z: {self.z}")
                self.getSubhaloData(sim, snapshot)
                self.getGroupData(sim, snapshot)
                self.getListGalaxyGroup()
                self.correctTheData()
                self.get_satellite_join_time(self.filtered_and_corrected_list_galaxy_groups, parallelize=True, rewrite=False) if snapshot == 99 else None
                self.saveListGalaxyGroup(additionalFileIdentifier)
                print(f"Finished processing for sim: {sim}, snapshot: {snapshot} of index {i}/{len(self.snapshot_dic.keys())}")
                
    def getListGalaxyGroup(self):
        headerInformation = {
            'simulation': self.sim,
            'description': self.simdata.get('description'),
            'boxsize_kpc': self.sim_boxsize_kpc,
            'hubble_param': self.simdata.get('hubble'),
            'redshift': self.z,
            'scale_factor': self.a,
        }
        
        list_of_galaxy_groups : ListGalaxyGroup = ListGalaxyGroup(headerInformation = headerInformation, listGalaxyGroups=[])

        temp_dict_galaxy_groups : dict[int, GalaxyGroup] = {} #temporary dictionary to hold galaxy groups while we build them since lookup is faster

        print(f"num subhalos: {self.subhaloGroupNum.shape[0]}")
        print(f"rough number of galaxygroups: {len(np.unique(self.subhaloGroupNum))}")
        for i in range(self.subhaloGroupNum.shape[0]):
            if self.flag[i] == False: #remove
                continue
            if np.isnan(self.SubhaloStellarPhotometrics[i][5]):
                continue
            group_num = self.subhaloGroupNum[i]
            
            if group_num > self.maxValidGroupIndex: ##NOTE assuming group numbers are sequential and start from 0
                break
            # if group_num not in temp_dic_validGroupMassIndexes:
            #     print(f"Skipping group number {group_num} as it does not meet mass criteria.")
            #     continue
            
            if group_num not in temp_dict_galaxy_groups: #check if galaxy group already exists - if not, create it - else just add the subhalo to it
                #initialize new empty galaxy group
                if self.groupMCrit200[group_num] < 1e13:
                    continue
                galaxyGroup = None
                galaxyGroup = GalaxyGroup(group_id=group_num, RCrit200=self.groupRCrit200[group_num], posCM=self.groupCM[group_num],  MCrit200=self.groupMCrit200[group_num], pos=self.groupPos[group_num], listSubhalos=[])
                # print(galaxyGroup.getNumSubhalos())
                temp_dict_galaxy_groups[group_num] = galaxyGroup
                
                list_of_galaxy_groups.addGalaxyGroup(galaxyGroup) #add to the master list
            # print(galaxyGroup.getNumSubhalos())
                
            # print(f"premass : {mass[i]}")
            subhalo = Subhalo(i, group_id=group_num, flag=self.flag[i], mass=self.mass[i], stellarMass=self.stellar_mass[i], groupNumber=self.subhaloGroupNum[i], position=self.subhaloPos[i], halfMassRad=self.subhaloHalfmassRad[i], vmaxRadius=self.SubhaloVmaxRad[i], luminosities=self.SubhaloStellarPhotometrics[i], luminositiesSDSS = self.SubhaloSDSSStellarPhotometrics[i], group_pos=self.groupPos[group_num]) #create subhalo    
            
            temp_dict_galaxy_groups[group_num].addSubhalo(subhalo) #add subhalo to the appropriate galaxy group

        print(f'Constructed ListGalaxyGroup with {list_of_galaxy_groups.getNumGalaxyGroups()} galaxy groups.')
        print(f' Average satellites: {list_of_galaxy_groups.getAverageNumSubhalosPerGalaxyGroup()}')
        self.list_of_galaxy_groups = list_of_galaxy_groups
    
    def correctTheData(self):
        print(f"WARNING.... double check: Correcting positions with boxsize: {self.sim_boxsize_kpc}")
        corrected_list_galaxy_groups = self.list_of_galaxy_groups.getCorrectedPositions(boxsize=self.sim_boxsize_kpc, parallelize=True)
        print(f'After correcting, ListGalaxyGroup has {corrected_list_galaxy_groups.getNumGalaxyGroups()} galaxy groups.')
        print(f' Average satellites: {corrected_list_galaxy_groups.getAverageNumSubhalosPerGalaxyGroup()}')
        
        # self.filtered_and_corrected_list_galaxy_groups = corrected_list_galaxy_groups.getFilterSubhalos(minGGMass=1e13, centralPosTolerance_kpc=None, M_r_max=-15, satWithinR200=True, parallelize=True)
        self.filtered_and_corrected_list_galaxy_groups = corrected_list_galaxy_groups.getFilterSubhalos(minGGMass=1e13, centralPosTolerance_kpc=None, satWithinR200=True, parallelize=True)
        print(f'After filtering, ListGalaxyGroup has {self.filtered_and_corrected_list_galaxy_groups.getNumGalaxyGroups()} galaxy groups.')
        print(f' Average satellites: {self.filtered_and_corrected_list_galaxy_groups.getAverageNumSubhalosPerGalaxyGroup()}')
        print(f' Range of satellites: {self.filtered_and_corrected_list_galaxy_groups.getRangeOfNumSubhalos()}')
        
    def saveListGalaxyGroup(self, additionalFileIdentifier:str=''):
        output_filename = self.scratchDataDirc + f'/galaxy_data_{self.sim}_{additionalFileIdentifier}.hdf5'
        with h5.File(output_filename, 'w') as f:
            # list_of_galaxy_groups.save_to_hdf5(f)
            self.filtered_and_corrected_list_galaxy_groups.save_to_hdf5(f, parallize=True)
        print(f'Saved galaxy data to {output_filename}')
        
        num_mostMassiveNotCentral=0
        for gg in self.filtered_and_corrected_list_galaxy_groups.listGalaxyGroups:
            if gg.getMostMassiveSubhalo().getIdx() != gg.getMostCentralSubhalo().getIdx():
                # print(f"GG ID: {gg.getGroupID()}, Num of galaxies: {gg.getNumSubhalos()}")
                num_mostMassiveNotCentral +=1
            # else:
            #     print(gg.getMostMassiveSubhalo().getPosition(), gg.getMostCentralSubhalo().getPosition())
        print(num_mostMassiveNotCentral)

    def getSubhaloData(self, sim, snapshot):
        self.flag = self.getSubhaloFlagData(sim, snapshot)
        self.mass, self.stellar_mass = self.getSubhaloMassData(sim, snapshot)
        self.subhaloGroupNum = self.getSubhaloGroupNumData(sim, snapshot)
        self.subhaloPos = self.getSubhaloPosData(sim, snapshot)
        self.subhaloHalfmassRad = self.getSubhaloHalfMassRadiusData(sim, snapshot)
        self.SubhaloVmaxRad = self.getSubhaloVMaxData(sim, snapshot)
        self.SubhaloStellarPhotometrics = self.getSubhaloLuminosityData(sim, snapshot)
        self.SubhaloSDSSStellarPhotometrics = self.getSubhaloSDSSLuminosityData(sim, snapshot)
        
        return self.flag, self.mass, self.stellar_mass, self.subhaloGroupNum, self.subhaloPos, self.subhaloHalfmassRad, self.SubhaloVmaxRad, self.SubhaloStellarPhotometrics, self.SubhaloSDSSStellarPhotometrics
        
    def getSubhaloFlagData(self, sim, snapshot):
        if not os.path.exists(self.scratchDataDirc + 'catalogs'):
            os.makedirs(self.scratchDataDirc + 'catalogs')
            print(f'created directory: {self.scratchDataDirc} "catalogs"')
        
        if not os.path.exists(self.scratchDataDirc + 'catalogs/SubhaloFlag'):
            os.makedirs(self.scratchDataDirc + 'catalogs/SubhaloFlag')
            print(f'created directory: {self.scratchDataDirc} "catalogs/SubhaloFlag"')
        flag=iapi_TNG.getSubhaloField('SubhaloFlag',simulation=sim,fileName=self.scratchDataDirc+'catalogs/SubhaloFlag/SubhaloFlag',snapshot=snapshot,rewriteFile=0)
        return flag

    def getSubhaloMassData(self, sim, snapshot):
        #let's fetch a field that will tell us about the mass of the galaxy
        #SubhaloMassType gives the total mass of all bound particles, separated by particle type
        if not os.path.exists(self.scratchDataDirc + 'catalogs/MassType'):
            os.makedirs(self.scratchDataDirc + 'catalogs/MassType')
            print(f'created directory: {self.scratchDataDirc} "catalogs/MassType"')
        mass=iapi_TNG.getSubhaloField('SubhaloMassType',simulation=sim,fileName=self.scratchDataDirc+'catalogs/MassType/MassType',snapshot=snapshot,rewriteFile=0) # in 10^10 solar masses/h
        print(mass.shape)

        #Pull the stellar mass: 
        print(mass[:,3])
        stellar_mass=mass[:,4]
        print(stellar_mass)

        stellar_mass=stellar_mass*10**10 * self.a / self.h #convert to one solar masses
        return mass, stellar_mass

    def getSubhaloGroupNumData(self, sim, snapshot):
        if not os.path.exists(self.scratchDataDirc + 'catalogs/SubhaloGrNr'):
            os.makedirs(self.scratchDataDirc + 'catalogs/SubhaloGrNr')
            print(f'created directory: {self.scratchDataDirc} "catalogs/SubhaloGrNr"')
        subhaloGroupNum = iapi_TNG.getSubhaloField('SubhaloGrNr',simulation = sim,fileName=self.scratchDataDirc+'catalogs/SubhaloGrNr/SubhaloGrNr',snapshot=snapshot,rewriteFile=0)
        return subhaloGroupNum
    
    def getSubhaloPosData(self, sim, snapshot):
        if not os.path.exists(self.scratchDataDirc + 'catalogs/SubhaloPos'):
            os.makedirs(self.scratchDataDirc + 'catalogs/SubhaloPos')
            print(f'created directory: {self.scratchDataDirc} "catalogs/SubhaloPos"')
        subhaloPos = iapi_TNG.getSubhaloField('SubhaloPos',simulation = sim,fileName=self.scratchDataDirc+'catalogs/SubhaloPos/SubhaloPos',snapshot=snapshot,rewriteFile=0) # in ckpc/h
        print(subhaloPos[0])
        ## convert positions from comoving to physical kpc
        subhaloPos = subhaloPos * self.a / self.h  # kpc
        print(subhaloPos[0])
        return subhaloPos
    
    def getSubhaloHalfMassRadiusData(self, sim, snapshot):
        if not os.path.exists(self.scratchDataDirc + 'catalogs/SubhaloHalfmassRad'):
            os.makedirs(self.scratchDataDirc + 'catalogs/SubhaloHalfmassRad')
            print(f'created directory: {self.scratchDataDirc} "catalogs/SubhaloHalfmassRad"')
        subhaloHalfmassRad = iapi_TNG.getSubhaloField('SubhaloHalfmassRad',simulation = sim,fileName=self.scratchDataDirc+'catalogs/SubhaloHalfmassRad/SubhaloHalfmassRad',snapshot=snapshot,rewriteFile=0) # in ckpc/h

        ## convert to kpc
        subhaloHalfmassRad = subhaloHalfmassRad * self.a / self.h  # kpc
        return subhaloHalfmassRad
        
    def getSubhaloVMaxData(self, sim, snapshot):
        if not os.path.exists(self.scratchDataDirc + 'catalogs/SubhaloVmaxRad'):
            os.makedirs(self.scratchDataDirc + 'catalogs/SubhaloVmaxRad')
            print(f'created directory: {self.scratchDataDirc} "catalogs/SubhaloVmaxRad"')
        SubhaloVmaxRad = iapi_TNG.getSubhaloField('SubhaloVmaxRad',simulation = sim,fileName=self.scratchDataDirc+'catalogs/SubhaloVmaxRad/SubhaloVmaxRad',snapshot=snapshot,rewriteFile=0) # in ckpc/h

        ## convert to kpc
        SubhaloVmaxRad = SubhaloVmaxRad * self.a / self.h  # kpc
        return SubhaloVmaxRad
    
    def getSubhaloLuminosityData(self, sim, snapshot):
        if not os.path.exists(self.scratchDataDirc + 'catalogs/SubhaloStellarPhotometrics'):
            os.makedirs(self.scratchDataDirc + 'catalogs/SubhaloStellarPhotometrics')
            print(f'created directory: {self.scratchDataDirc} "catalogs/SubhaloStellarPhotometrics"')

        rewriteFile=0
        fileName=self.scratchDataDirc+'catalogs/SubhaloStellarPhotometrics/SubhaloStellarPhotometrics'
            
            
        SubhaloStellarPhotometrics = iapi_TNG.getSubhaloField('SubhaloStellarPhotometrics',simulation = sim,fileName=self.scratchDataDirc+'catalogs/SubhaloStellarPhotometrics/SubhaloStellarPhotometricsDefault',snapshot=snapshot,rewriteFile=0) # Eight bands: U, B, V, K, g, r, i, z, all in mag
        return SubhaloStellarPhotometrics

    def getSubhaloSDSSLuminosityData(self, sim, snapshot):
        if not os.path.exists(self.scratchDataDirc + 'catalogs/SubhaloStellarPhotometrics'):
            os.makedirs(self.scratchDataDirc + 'catalogs/SubhaloStellarPhotometrics')
            print(f'created directory: {self.scratchDataDirc} "catalogs/SubhaloStellarPhotometrics"')

        if sim == 'TNG-Cluster':
            return np.zeros(len(self.SubhaloStellarPhotometrics))

        rewriteFile=0
        fileName=self.scratchDataDirc+'catalogs/SubhaloStellarPhotometrics/SubhaloSDSSStellarPhotometrics'
            
        if not os.path.exists(fileName+'.hdf5') or rewriteFile==1:
            # "http://www.tng-project.org/api/TNG300-1/files/stellar_photometry.99.hdf5"
            url='http://www.tng-project.org/api/'+sim+'/files/stellar_photometry.'+str(snapshot)+'.hdf5'
            dataFile=iapi_TNG.get(url,fName=fileName)
        elif rewriteFile == 0:
            dataFile = fileName + '.hdf5'
            
        with h5.File(dataFile,'r') as f:
            print(f.keys())
            SubhaloSDSSStellarPhotometrics=f['Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc'][:]
            print(SubhaloSDSSStellarPhotometrics.shape)
            # print(SubhaloStellarPhotometrics[0:5,:])
            # print(SubhaloVmaxRad.shape)
            # low redshifts: sdss_u, sdss_g, sdss_r, sdss_i, sdss_z, wfc_acs_f606w, des_y, jwst_f150w
            # high redshifts: sdss_u, sdss_g, sdss_r, sdss_i, sdss_z (at redshift 1.5 aka snapshot 40)
        SubhaloSDSSStellarPhotometricsTrimmed = []
        for sdss in SubhaloSDSSStellarPhotometrics:
            SubhaloSDSSStellarPhotometricsTrimmed.append((sdss[0][0], sdss[1][0], sdss[2][0], sdss[3][0],sdss[4][0],sdss[5][0],sdss[6][0],sdss[7][0]))
            
        return SubhaloSDSSStellarPhotometricsTrimmed
        
    def getGroupData(self, sim, snapshot):
        self.groupMCrit200 = self.getGroupMCrit200Data(sim, snapshot)
        self.groupRCrit200 = self.getGroupRCrit200Data(sim, snapshot)
        self.groupCM = self.getGroupCMData(sim, snapshot)
        self.groupPos = self.getGroupPosData(sim, snapshot)
        
        return self.groupMCrit200, self.groupRCrit200, self.groupCM, self.groupPos
    
    def getGroupMCrit200Data(self, sim, snapshot):
        if not os.path.exists(self.scratchDataDirc + 'catalogs/GroupMCrit200'):
            os.makedirs(self.scratchDataDirc + 'catalogs/GroupMCrit200')
            print(f'created directory: {self.scratchDataDirc} "catalogs/GroupMCrit200"')
        groupMCrit200 = iapi_TNG.getHaloField('Group_M_Crit200',simulation = sim,fileName=self.scratchDataDirc+'catalogs/GroupMCrit200/GroupMCrit200',snapshot=snapshot,rewriteFile=0) # in 10^10 solar masses/h

        ## convert to solar masses
        groupMCrit200 = groupMCrit200 * 1e10 * self.a / self.h  # solar masses

        validGroupMassIndexes = np.where(groupMCrit200 > 1e13)[0]
        print(validGroupMassIndexes)
        print(f'Number of galaxy groups with M200 > 1e13 solar masses: {np.sum(validGroupMassIndexes)}')
        # validGroupMCrit200 = groupMCrit200[validGroupMassIndexes]
        # temp_dic_validGroupMassIndexes = {index: True for index in validGroupMassIndexes} #create a temporary dictionary for faster lookup
        maxValidGroupIndex = np.max(validGroupMassIndexes)
        print(f'Max valid group index: {maxValidGroupIndex}')
        
        self.maxValidGroupIndex = maxValidGroupIndex
        return groupMCrit200
        
    def getGroupRCrit200Data(self, sim, snapshot):
        if not os.path.exists(self.scratchDataDirc + 'catalogs/GroupRCrit200'):
            os.makedirs(self.scratchDataDirc + 'catalogs/GroupRCrit200')
            print(f'created directory: {self.scratchDataDirc} "catalogs/GroupRCrit200"')
        groupRCrit200 = iapi_TNG.getHaloField('Group_R_Crit200',simulation = sim,fileName=self.scratchDataDirc+'catalogs/GroupRCrit200/GroupRCrit200',snapshot=snapshot,rewriteFile=0) # in ckpc/h

        print(groupRCrit200[0])
        ## convert to kpc
        groupRCrit200 = groupRCrit200 * self.a / self.h  # kpc
        print(groupRCrit200[0])
        return groupRCrit200
        
    def getGroupCMData(self, sim, snapshot):
        if not os.path.exists(self.scratchDataDirc + 'catalogs/GroupCM'):
            os.makedirs(self.scratchDataDirc + 'catalogs/GroupCM')
            print(f'created directory: {self.scratchDataDirc} "catalogs/GroupCM"')
        groupCM = iapi_TNG.getHaloField('GroupCM',simulation = sim,fileName=self.scratchDataDirc+'catalogs/GroupCM/GroupCM',snapshot=snapshot,rewriteFile=0) # in ckpc/h

        ## convert to kpc
        groupCM = groupCM * self.a / self.h  # kpc

        # select only validGroupMassIndexes
        # validGroupCM = groupCM[validGroupMassIndexes]
        return groupCM
        
    def getGroupPosData(self, sim, snapshot):
        if not os.path.exists(self.scratchDataDirc + 'catalogs/GroupPos'):
            os.makedirs(self.scratchDataDirc + 'catalogs/GroupPos')
            print(f'created directory: {self.scratchDataDirc} "catalogs/GroupPos"')
        groupPos = iapi_TNG.getHaloField('GroupPos',simulation = sim,fileName=self.scratchDataDirc+'catalogs/GroupPos/GroupPos',snapshot=snapshot,rewriteFile=0) # in ckpc/h

        ## convert to kpc
        groupPos = groupPos * self.a / self.h  # kpc

        # select only validGroupMassIndexes
        # validGroupPos = groupPos[validGroupMassIndexes]
        return groupPos
    
    
    # def get_satellite_join_time(self, listGalaxyGroups : ListGalaxyGroup, rewrite:Optional[bool] = None):
    #     #Box boundary 
    #     # L=75000. #kpc
    #     # halfbox=L/2.   
    #     # h=0.6774 
    #     joinTime = JoinTime(self.sim, self.snapshot)
    #     if rewrite is None:
    #         # rewrite = self.generalRewrite
    #         rewrite=True

    #     totalSatellites = sum(gg.getNumSubhalos() for gg in listGalaxyGroups.getAllGalaxyGroups())
    #     processedSatelliteIds = []
    #     print(f"Total number of satellites to process: {totalSatellites}")

    #     processedJoinTimes = []
        
    #     satellitesWihtoutMergerTree = []
    #     #load satellitesWihtoutMergerTree from file if exists
    #     # if rewrite==True:
    #     if os.path.exists(self.scratchPlotDirc + f'/satellites_without_merger_tree_{self.sim}_{self.snapshot_dic[self.snapshot][1]}.txt'):
    #         with open(self.scratchPlotDirc + f'/satellites_without_merger_tree_{self.sim}_{self.snapshot_dic[self.snapshot][1]}.txt', 'r') as f:
    #             lines = f.readlines()
    #             for line in lines[1:]:  # Skip header line
    #                 parts = line.split('\t')
    #                 if len(parts) > 0:
    #                     satellitesWihtoutMergerTree.append(int(parts[0]))
    #         print(f"Loaded {len(satellitesWihtoutMergerTree)} satellites without merger tree from file.")
    #     else:
    #         print("No existing file for satellites without merger tree found, starting with an empty list.")
    #         #prep the file with header
    #         with open(self.scratchPlotDirc + f'/satellites_without_merger_tree_{self.sim}_{self.snapshot_dic[self.snapshot][1]}.txt', 'w') as f:
    #             f.write("SatelliteSubhaloID\n")
        
    #     #get all join times
    
    #     if not rewrite:
    #         #check if file already exists, if so, skip
    #         if os.path.exists(self.scratchPlotDirc + f'/join_times_and_parameter_changes_{self.sim}_{self.snapshot_dic[self.snapshot][1]}.txt'):
    #             print(f"File join_times_and_parameter_changes_{self.sim}_{self.snapshot_dic[self.snapshot][1]}.txt already exists, skipping...")
    #     else:
    #         with open(self.scratchPlotDirc + f'/join_times_and_parameter_changes_{self.sim}_{self.snapshot_dic[self.snapshot][1]}.txt', 'w') as f:
    #             f.write("GalaxyGroupID\tSubhaloID\tNumMembers\tClusterMass\tJoiningRedshift\tSeparationAtZ0\tSeparationNormAtZ0\tDeltaGasMass\tDeltaTotalMass\tDeltaDMMass\tDeltaStellarMass\tDeltaVelSq\tJoiningSnap\tClosestApproach\tClosestApproachNorm\tClosestApproachRedshift\tJoinProgID\tDeltaAngularMomentum\tSatelliteMassAtJoining\tHostProgID\n")
    #             for gg in listGalaxyGroups.getAllGalaxyGroups():
    #                 # print(f"Processing Galaxy Group ID: {gg.getGroupID()}")
    #                 gg_id = gg.getGroupID()
    #                 num_members = gg.getNumSubhalos()
    #                 cluster_mass = gg.getMCrit200()
    #                 #get the subhalo ID of the central galaxy, which is the one that joins the host halo
    #                 central_subhalo : GalaxyGroup = gg.getCentralSubhalo()
    #                 for i, subhalo in enumerate(gg.getSatelliteSubhalos()):
    #                     if central_subhalo is not None:
    #                         # print(f"id: {subhalo.getIdx()}")
    #                         #check if satellite has no merger tree
    #                         if subhalo.getIdx() in satellitesWihtoutMergerTree:
    #                             print(f"Skipping subhalo {subhalo.getIdx()} (previously identified as having no merger tree)")
    #                             continue
    #                         print(f"Processing subhalo {subhalo.getIdx()}, progress: {i}/{num_members-1} satellites in this group, total progress: {len(processedSatelliteIds)}/{totalSatellites} satellites", end='\r', flush=True)
    #                         join_time_info = joinTime.computeJoinTimes(hostID=central_subhalo.getIdx(), ID=subhalo.getIdx(), L=self.sim_boxsize_kpc, halfbox=self.sim_boxsize_kpc/2, fname=self.scratchDataDirc+'/mergerTree')
    #                         processedSatelliteIds.append(subhalo.getIdx())
    #                         if join_time_info is None:
    #                             print(f"Skipping subhalo {subhalo.getIdx()} (no merger tree available)")
    #                             satellitesWihtoutMergerTree.append(subhalo.getIdx())
    #                             #write into file
    #                             with open(self.scratchPlotDirc + f'/satellites_without_merger_tree_{self.sim}_{self.snapshot_dic[self.snapshot][1]}.txt', 'a') as g:
    #                                 g.write(f"{subhalo.getIdx()}\n")
    #                             continue
    #                         f.write(f"{gg_id}\t{subhalo.getIdx()}\t{num_members}\t{cluster_mass}\t{join_time_info[0]}\t{join_time_info[1]}\t{join_time_info[2]}\t{join_time_info[3]}\t{join_time_info[4]}\t{join_time_info[5]}\t{join_time_info[6]}\t{join_time_info[7]}\t{join_time_info[8]}\t{join_time_info[9]}\t{join_time_info[10]}\t{join_time_info[11]}\t{join_time_info[12]}\t{join_time_info[13]}\t{join_time_info[14]}\n")
    #                         processedJoinTimes.append(join_time_info[0])
    #                         subhalo.setJoiningRedshiftInfo(join_time_info)
    #                         # if i == 0:
    #                             # print(f"jointime: {join_time_info[0]}")
    #     print(f"\nFinished processing all satellites. Total processed: {len(processedSatelliteIds)}, expected: {totalSatellites}. Satellites without merger tree: {len(satellitesWihtoutMergerTree)}, with non nan join times: {len(np.where(np.isfinite(processedJoinTimes)))}/{len(processedJoinTimes)}")
    
    
    
    
    
    def get_satellite_join_time(
        self,
        listGalaxyGroups: ListGalaxyGroup,
        rewrite: Optional[bool] = None,
        parallelize: bool = False,
        n_processes: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ):
        #Box boundary 
        # L=75000. #kpc
        # halfbox=L/2.   
        # h=0.6774 
        if rewrite is None:
            # rewrite = self.generalRewrite
            rewrite=True

        join_times_path = self.scratchPlotDirc + f'/join_times_and_parameter_changes_{self.sim}_{self.snapshot_dic[self.snapshot][1]}.txt'
        satellites_without_tree_path = self.scratchPlotDirc + f'/satellites_without_merger_tree_{self.sim}_{self.snapshot_dic[self.snapshot][1]}.txt'

        join_times_header = (
            "GalaxyGroupID\tSubhaloID\tNumMembers\tClusterMass\tJoiningRedshift\tSeparationAtZ0\tSeparationNormAtZ0\t"
            "DeltaGasMass\tDeltaTotalMass\tDeltaDMMass\tDeltaStellarMass\tDeltaVelSq\tJoiningSnap\tClosestApproach\t"
            "ClosestApproachNorm\tClosestApproachRedshift\tJoinProgID\tDeltaAngularMomentum\tSatelliteMassAtJoining\tHostProgID\n"
        )

        totalSatellites = sum(gg.getNumSubhalos() for gg in listGalaxyGroups.getAllGalaxyGroups())
        print(f"Total number of satellites to process (including centrals in counts): {totalSatellites}")

        processedJoinTimes = []
        
        satellitesWihtoutMergerTree: set[int] = set()
        #load satellitesWihtoutMergerTree from file if exists
        # if rewrite==True:
        if os.path.exists(satellites_without_tree_path):
            with open(satellites_without_tree_path, 'r') as f:
                for line in f:
                    s = line.strip()
                    if not s or s.lower().startswith('satellitesubhaloid'):
                        continue
                    try:
                        satellitesWihtoutMergerTree.add(int(s.split('\t')[0]))
                    except Exception:
                        continue
            print(f"Loaded {len(satellitesWihtoutMergerTree)} satellites without merger tree from file.")
        else:
            print("No existing file for satellites without merger tree found, starting with an empty list.")
            #prep the file with header
            with open(satellites_without_tree_path, 'w') as f:
                f.write("SatelliteSubhaloID\n")
        
        # If not rewriting, resume from existing join-times file when available.
        completed_satellite_ids: set[int] = set()
        join_file_mode = 'w'
        should_write_header = True
        if not rewrite and os.path.exists(join_times_path):
            join_file_mode = 'a'
            should_write_header = os.path.getsize(join_times_path) == 0
            try:
                with open(join_times_path, 'r') as f:
                    for line in f:
                        s = line.strip()
                        if not s or s.lower().startswith('galaxygroupid'):
                            continue
                        parts = s.split('\t')
                        if len(parts) < 2:
                            continue
                        try:
                            completed_satellite_ids.add(int(parts[1]))
                        except Exception:
                            continue
                print(
                    f"Resuming from existing {os.path.basename(join_times_path)}: "
                    f"{len(completed_satellite_ids)} satellites already completed."
                )
            except Exception as e:
                print(f"WARNING: Failed to read existing join-times file for resume ({e}); will append anyway.")

        if rewrite:
            join_file_mode = 'w'
            should_write_header = True

        # Build tasks (one per satellite)
        tasks: list[tuple] = []
        for gg in listGalaxyGroups.getAllGalaxyGroups():
            gg_id = gg.getGroupID()
            num_members = gg.getNumSubhalos()
            cluster_mass = gg.getMCrit200()
            central_subhalo = gg.getCentralSubhalo()
            if central_subhalo is None:
                continue

            host_id = central_subhalo.getIdx()
            for subhalo in gg.getSatelliteSubhalos():
                sat_id = subhalo.getIdx()
                if sat_id in satellitesWihtoutMergerTree:
                    continue
                if sat_id in completed_satellite_ids:
                    continue
                tasks.append(
                    (
                        self.sim,
                        self.snapshot,
                        host_id,
                        sat_id,
                        self.sim_boxsize_kpc,
                        self.sim_boxsize_kpc/2,
                        self.scratchDataDirc + '/mergerTree',
                        gg_id,
                        num_members,
                        cluster_mass,
                    )
                )

        if len(tasks) == 0:
            if completed_satellite_ids:
                print("No satellites left to process (resume file already contains all targets).")
            else:
                print("No satellites to process (after filtering).")
            return

        if parallelize:
            if n_processes is None:
                try:
                    # Prefer the module-level import when available
                    n_processes = get_optimal_processes(len(tasks))
                except NameError:
                    # Notebook / partial-import safety
                    from myproject.utilities.parallelTools import get_optimal_processes as _get_optimal_processes

                    n_processes = _get_optimal_processes(len(tasks))
                except Exception:
                    n_processes = max(1, mp.cpu_count() - 1)
            if chunk_size is None:
                chunk_size = max(1, len(tasks) // (n_processes * 8))
            print(f"Computing join times in parallel: {len(tasks)} satellites, {n_processes} processes, chunksize={chunk_size}")
        else:
            print(f"Computing join times serially: {len(tasks)} satellites")

        newly_missing_tree: list[int] = []
        completed = 0
        total = len(tasks)

        with open(join_times_path, join_file_mode) as f:
            if should_write_header:
                f.write(join_times_header)

            if parallelize:
                with mp.Pool(processes=n_processes) as pool:
                    for result in pool.imap_unordered(_compute_join_time_worker, tasks, chunksize=chunk_size):
                        completed += 1
                        if completed % max(1, total // 200) == 0 or completed == total:
                            pct = 100.0 * completed / total
                            print(f"\rProgress: {completed}/{total} ({pct:.1f}%)", end='', flush=True)

                        sat_id, gg_id, num_members, cluster_mass, join_time_info = result
                        if join_time_info is None:
                            newly_missing_tree.append(sat_id)
                            continue
                        f.write(
                            f"{gg_id}\t{sat_id}\t{num_members}\t{cluster_mass}\t"
                            + "\t".join(str(x) for x in join_time_info)
                            + "\n"
                        )
                        processedJoinTimes.append(join_time_info[0])
            else:
                for task in tasks:
                    sat_id, gg_id, num_members, cluster_mass, join_time_info = _compute_join_time_worker(task)
                    completed += 1
                    if completed % max(1, total // 200) == 0 or completed == total:
                        pct = 100.0 * completed / total
                        print(f"\rProgress: {completed}/{total} ({pct:.1f}%)", end='', flush=True)

                    if join_time_info is None:
                        newly_missing_tree.append(sat_id)
                        continue
                    f.write(
                        f"{gg_id}\t{sat_id}\t{num_members}\t{cluster_mass}\t"
                        + "\t".join(str(x) for x in join_time_info)
                        + "\n"
                    )
                    processedJoinTimes.append(join_time_info[0])

        print()  # newline after progress

        # Update satellites-without-tree file in one batch (much faster)
        if newly_missing_tree:
            with open(satellites_without_tree_path, 'a') as g:
                for sid in newly_missing_tree:
                    g.write(f"{sid}\n")
            satellitesWihtoutMergerTree.update(newly_missing_tree)

        print(
            f"Finished processing {total} satellites. "
            f"Newly missing merger tree: {len(newly_missing_tree)}. "
            f"Total missing-tree cache size: {len(satellitesWihtoutMergerTree)}. "
            f"Finite join redshifts: {int(np.sum(np.isfinite(processedJoinTimes)))}/{len(processedJoinTimes)}"
        )



def _compute_join_time_worker(args: tuple):
    """Worker to compute a single satellite join-time record.

    Returns:
        (sat_id, gg_id, num_members, cluster_mass, join_time_info_or_None)

    Note: JoinTime object is instantiated inside the worker for pickling safety.
    """
    sim, snapshot, host_id, sat_id, L, halfbox, fname, gg_id, num_members, cluster_mass = args
    joinTime = JoinTime(sim, snapshot)
    join_time_info = joinTime.computeJoinTimes(hostID=host_id, ID=sat_id, L=L, halfbox=halfbox, fname=fname)
    if join_time_info is None:
        return sat_id, gg_id, num_members, cluster_mass, None
    try:
        join_time_list = list(join_time_info)
    except Exception:
        join_time_list = [join_time_info]
    return sat_id, gg_id, num_members, cluster_mass, join_time_list