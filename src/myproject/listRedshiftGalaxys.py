#AP 2026

from myproject.redshiftGalaxyAnalysis import GalaxyAnalysis
from myproject.utilities.snapshotEnum import SnapshotEnum

class ListRedshiftGalaxy:
    def __init__(self):
        self.galaxy_dic : dict[str, GalaxyAnalysis] = {} #dictionary of redshift to GalaxyAnalysis

    def getGalaxyAnalysisSnapshot(self, snapshot_enum : str) -> GalaxyAnalysis:
        if snapshot_enum not in self.galaxy_dic:
            self.galaxy_dic[snapshot_enum] = GalaxyAnalysis(snapshot_enum)
        return self.galaxy_dic[snapshot_enum]
    
    def getAllGalaxyAnalysis(self):
        return self.galaxy_dic.values()

    def getAllLoadedSnapshots(self):
        return list(self.galaxy_dic.keys())
    
    def addGalaxyAnalysis(self, snapshot_enum : str, galaxy_analysis : GalaxyAnalysis):
        self.galaxy_dic[snapshot_enum] = galaxy_analysis
        
    def initializeAllGalaxyAnalysis(self, snapshot_enums : list, simulation:str, luminosityType:str, generalErrorbar:str = 'poisson', generalRewrite:bool = True):
        if luminosityType not in ['SDSS', 'Default']:
            raise ValueError("Invalid luminosity type. Must be 'SDSS' or 'Default'.")
        if simulation not in ['TNG300-1', 'TNG-Cluster', 'TNG300-1, TNG-Cluster', 'TNG-Cluster, TNG300-1']:
            raise ValueError("Invalid simulation. Must be 'TNG300-1' or 'TNG-Cluster' or 'TNG300-1, TNG-Cluster' or 'TNG-Cluster, TNG300-1'.")
        for snapshot_enum in snapshot_enums:
            galaxy_analysis = GalaxyAnalysis(sim=simulation, snapshot=snapshot_enum, generalErrorbar=generalErrorbar, generalRewrite=generalRewrite, luminosityType=luminosityType)
            self.addGalaxyAnalysis(snapshot_enum, galaxy_analysis)