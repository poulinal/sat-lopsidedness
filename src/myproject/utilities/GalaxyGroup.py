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
    
        self.listSubhalos = listSubhalos
        # print(f"self.listSubhalos: {len(self.listSubhalos)}")
        self.lenSubhalos = len(self.listSubhalos)
        
    def addSubhalo(self, subhalo : Subhalo):
        self.listSubhalos.append(subhalo)
        self.lenSubhalos += 1
        
    def getNumSubhalos(self):
        return self.lenSubhalos
    
    def getSubhalos(self) -> list[Subhalo]:
        return self.listSubhalos
    
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
    
    def setPosCM(self, newPosCM : np.ndarray):
        self.posCM = newPosCM
        
    def setPos(self, newPos : np.ndarray):
        self.pos = newPos
        
    def setRCrit200(self, newRCrit200 : float):
        self.RCrit200 = newRCrit200
    
    
    