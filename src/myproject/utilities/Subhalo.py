# ADP 2026

class Subhalo:

    def __init__(self, idx, flag, mass, stellarMass, groupNumber, position, halfMassRad, vmaxRadius, luminosities):
        self.idx = idx
        self.flag = flag
        self.mass = mass
        self.stellarMass = stellarMass
        self.groupNumber = groupNumber
        self.position = position
        self.halfMassRad = halfMassRad
        self.vmaxRadius = vmaxRadius
        self.luminosities = luminosities
        
    def getIdx(self):
        return self.idx
    
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
        return self.luminosities