# ADP 2026

from myproject.utilities.Subhalo import Subhalo
import numpy as np

class GalaxyGroup:
    def __init__(self, group_id, RCrit200, MCrit200, posCM, pos, listSubhalos : list[Subhalo]=[]):
        self.group_id = group_id
        self.RCrit200 = RCrit200
        self.MCrit200 = MCrit200
        self.posCM = posCM
        self.pos = pos
        
        self.listSatelliteSubhalos = []
        self.centralSubhalo = None
    
        self.listSubhalos = self.setSubhaloList(listSubhalos)
        
    def addSubhalo(self, subhalo : Subhalo):
        self.listSubhalos.append(subhalo)
        self.lenSubhalos += 1
        
        # Re-identify central and satellite subhalos
        max_mass = -1
        central_subhalo = None
        satellite_subhalos = []
        for sh in self.listSubhalos:
            if sh.getMass() > max_mass:
                max_mass = sh.getMass()
                central_subhalo = sh
        for sh in self.listSubhalos:
            if sh != central_subhalo:
                satellite_subhalos.append(sh)
        self.setCentralSubhalo(central_subhalo)
        self.setSatelliteSubhalos(satellite_subhalos)
        
    def getNumSubhalos(self):
        return self.lenSubhalos
    
    def getSubhalos(self) -> list[Subhalo]:
        return self.listSubhalos

    def getSatelliteSubhalos(self) -> list[Subhalo]:
        # Return only satellite subhalos (exclude central which has pos (0, 0, 0))
        return self.listSatelliteSubhalos
    
    def getCentralSubhalo(self) -> Subhalo:
        return self.centralSubhalo
    
    def getGroupID(self):
        return self.group_id
    
    def getRCrit200(self):
        return self.RCrit200
    
    def getMCrit200(self):
        return self.MCrit200
    
    def getPosCM(self):
        return self.posCM
    
    def getPos(self):
        return self.pos
    
    def getSubhaloI(self, i):
        return self.listSubhalos[i]
    
    def setSubhaloList(self, newListSubhalos : list[Subhalo]):
        self.listSubhalos = newListSubhalos
        self.lenSubhalos = len(newListSubhalos)
        
        # Identify central (largest subhalo in group) and satellite subhalos
        max_mass = -1
        central_subhalo = None
        satellite_subhalos = []
        for subhalo in newListSubhalos:
            if subhalo.getMass() > max_mass:
                max_mass = subhalo.getMass()
                central_subhalo = subhalo
        for subhalo in newListSubhalos:
            if subhalo != central_subhalo:
                satellite_subhalos.append(subhalo)
        self.setCentralSubhalo(central_subhalo)
        self.setSatelliteSubhalos(satellite_subhalos)
        
        return newListSubhalos
    
    def setCentralSubhalo(self, central_subhalo : Subhalo):
        self.centralSubhalo = central_subhalo
        
    def setSatelliteSubhalos(self, satellite_subhalos : list[Subhalo]):
        self.listSatelliteSubhalos = satellite_subhalos
    
    def setPosCM(self, newPosCM : np.ndarray):
        self.posCM = newPosCM
        
    def setPos(self, newPos : np.ndarray):
        self.pos = newPos
        
    def setRCrit200(self, newRCrit200 : float):
        self.RCrit200 = newRCrit200
    
    def getSubhaloByID(self, subhalo_id : int) -> Subhalo | None:
        for subhalo in self.listSubhalos:
            if subhalo.getIdx() == subhalo_id:
                return subhalo
        # print("Couldnt find subhalo idx")
        return None
    
    