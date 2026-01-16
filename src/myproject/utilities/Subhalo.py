# ADP 2026

import numpy as np

class Subhalo:

    def __init__(self, idx : int, group_id : int, flag : int, mass : float, stellarMass : float, groupNumber : int, position : np.ndarray, halfMassRad : float, vmaxRadius : float, luminosities : np.ndarray[1,8]):
        """
        Constructor for Subhalo class.

        Args:
            idx (int): _unique subhalo index_
            group_id (int): _ID of the parent galaxy group_
            flag (int): _subhalo flag (0 = non-cosmological origin, 1 = cosmological origin)_
            mass (float): _mass in Msun_
            stellarMass (float): _stellar mass in Msun_
            groupNumber (int): _index of the subhalo within its parent group_
            position (np.ndarray): _position as a numpy array with 3 elements (x, y, z) in kpc_
            halfMassRad (float): _Radius enclosing half the mass in kpc_
            vmaxRadius (float): _Radius at which the maximum circular velocity is reached in kpc_
            luminosities (np.ndarray[1,8]): _array with 8 elements corresponding to U, B, V, K, g, r, i, z bands. Units: mag_
        """
        self.idx = idx # unique subhalo index
        self.group_id = group_id # ID of the parent galaxy group
        self.flag = flag # subhalo flag (0 = non-cosmological origin, 1 = cosmological origin)
        self.mass = mass # in Msun
        self.stellarMass = stellarMass # in Msun
        self.groupNumber = groupNumber # Index of the subhalo within its parent group
        self.position = position # np.ndarray with 3 elements (x, y, z) in kpc
        self.halfMassRad = halfMassRad # in kpc
        self.vmaxRadius = vmaxRadius # in kpc
        self.luminosities = luminosities # array with 8 elements corresponding to U, B, V, K, g, r, i, z bands. Units: mag
        
    def getIdx(self):
        return self.idx
    
    def getGroupID(self):
        return self.group_id
    
    def getFlag(self):
        return self.flag
    
    def getMass(self):
        return self.mass
    
    def getStellarMass(self):
        return self.stellarMass
    
    def getGroupNumber(self):
        return self.groupNumber
    
    def getPosition(self):
        return self.position
    
    def getHalfMassRad(self):
        return self.halfMassRad
    
    def getVmaxRadius(self):
        return self.vmaxRadius
    
    def getLuminosities(self):
        return self.luminosities # returns array with 8 elements corresponding to U, B, V, K, g, r, i, z bands
    
    def getRbandMagnitude(self):
        return self.getLuminosities()[5]
    
    def setPosition(self, newPosition : np.ndarray):
        self.position = newPosition