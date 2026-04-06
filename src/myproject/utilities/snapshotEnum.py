# AP 2026

from enum import Enum

class SnapshotEnum(Enum):
    SNAPSHOT_99 = (99, 0, 'z0p0')
    SNAPSHOT_91 = (91, 0.1, 'z0p1')
    SNAPSHOT_84 = (84, 0.2, 'z0p2')
    SNAPSHOT_78 = (78, 0.3, 'z0p3')
    SNAPSHOT_72 = (72, 0.4, 'z0p4')
    SNAPSHOT_67 = (67, 0.5, 'z0p5')
    SNAPSHOT_59 = (59, 0.7, 'z0p7')
    SNAPSHOT_50 = (50, 1.0, 'z1p0')
    SNAPSHOT_40 = (40, 1.5, 'z1p5')
    SNAPSHOT_33 = (33, 2.0, 'z2p0')
    SNAPSHOT_25 = (25, 3.0, 'z3p0')
    
    def getAllSnapshots():
        """Return a list of snapshot tuples sorted by redshift (ascending).

        Each item is the tuple stored in the enum value, e.g. (99, 0.0, 'z0p0').
        Sorting ensures plotting and iteration proceed from low to high redshift.
        """
        return sorted([snapshot.value for snapshot in SnapshotEnum], key=lambda v: v[1])
    
    def getAllRedshifts():
        """Return a list of redshift values sorted in ascending order."""
        return sorted([snapshot.value[1] for snapshot in SnapshotEnum])
    
        
    