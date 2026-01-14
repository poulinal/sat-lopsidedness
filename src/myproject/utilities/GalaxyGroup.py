# ADP 2026

from myproject.utilities.Subhalo import Subhalo

class GalaxyGroup:
    def __init__(self, group_id, MCrit200, posCM, pos, listSubhalos : list[Subhalo]=[]):
        self.group_id = group_id
        self.MCrit200 = MCrit200
        self.posCM = posCM
        self.pos = pos
    
        self.listSubhalos = []
        
    def addSubhalo(self, subhalo : Subhalo):
        self.listSubhalos.append(subhalo)
        
    def getNumSubhalos(self):
        return len(self.listSubhalos)
    
    def getSubhalos(self):
        return self.listSubhalos
    
    def getGroupID(self):
        return self.group_id
    
    def getMCrit200(self):
        return self.MCrit200
    
    def getPosCM(self):
        return self.posCM
    
    def getPos(self):
        return self.pos
    
    def getSubhaloI(self, i):
        return self.listSubhalos[i]
    
    
    