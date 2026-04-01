from myproject.utilities.iapi_TNG import get
import os
import h5py as h5
import numpy as np
from requests.exceptions import HTTPError
import illustris_python as il

class JoinTime():
    def __init__(self, sim, snapshot):
        self.sim=sim
        self.snapshot=snapshot

        #Administrative URL pulling
        baseUrl = 'http://www.tng-project.org/api/'
        # headers = {"api-key":"env{apikey}"}
        r=get(baseUrl)
        names = [sim['name'] for sim in r['simulations']]
        i = names.index(self.sim)
        sim = get( r['simulations'][i]['url'] )
        self.snaps = get( sim['snapshots'] )

    #also pulled from join time code
    def getsub(self, snapnum,subid):
        #Pull url of a sub
        #subid is the ID back into the subhalo group catalog
        url= f'https://www.tng-project.org/api/{self.sim}/snapshots/'+str(snapnum)+'/subhalos/'+str(subid)+'/'
        sub=get(url)
        return(sub)

    def gettree(self, subid, fname:str=''):
        #pull the z=0 merger tree for a subhlao
        #some subhalos don't have merger trees
        fname = fname+'/sublink_mpb_'+str(subid) if fname != '' else ''
        fName = 'Trees/sublink_mpb_'+str(subid) if fname == '' else fname
        if os.path.exists(fName+'.hdf5'):
            return(fName+'.hdf5')
        else:
            print(f"file does not exist: {fName}.hdf5")
        url=f'https://www.tng-project.org/api/{self.sim}/snapshots/{self.snapshot}/subhalos/'+str(subid)+'/sublink/mpb.hdf5'
        tree=get(url,fName=fName)
        return(tree)

    def getredshift(self, snapnum):
        #convert a snapshot number to a redshift
        return(self.snaps[snapnum]['redshift'])
        
    def computeJoinTimes(self, ID,hostID, L, halfbox, fname:str):
        """
        Use the satellite and host trees to find the joining redshift of a satellite based on when it first approached its z=0 FoF group within 3R200
        Identify the change in satellite parameters since they joined

        Returns:        
        joinred: redshift at which the satellite joined its current host halo
        sep_z0: separation between satellite and host at z=0 in kpc
        sep_norm: separation at z=0 normalized by the host's R200
        del_M: change in satellite gas mass since joining in Msun
        del_T_all: change in satellite kinetic energy since joining, using all mass in Msun*(km/s)^2
        del_T_lim: change in satellite kinetic energy since joining, using only gas+star mass in Msun*(km/s)^2
        del_M_total: change in satellite total mass since joining in Msun
        del_M_dm: change in satellite dark matter mass since joining in Msun
        del_M_stars: change in satellite stellar mass since joining in Msun
        del_vsq: change in satellite relative velocity squared since joining in (km/s)^2
        joinsnap: snapshot number at which the satellite joined its current host halo
        closest: closest approach between satellite and host in kpc
        closest_norm: closest approach normalized by host R200
        closest_z: redshift at which the closest approach occurred
        joinprog: ID of the progenitor galaxy of the subhalo at the joining snapshot, used to pull progenitor cutout in other code
        del_L: change in satellite angular momentum since joining in Msun*kpc*km/s
        s_mass_j: satellite stellar mass at joining in Msun
        hostprog_ID: ID of the progenitor galaxy of the host halo at the joining snapshot, used to pull progenitor cutout in other code
        L_join: satellite angular momentum at joining in Msun*kpc*km/s
        L_0: satellite angular momentum at z=0 in Msun*kpc*km/s
        first1R200
        """


        def _as_1d(x):
            return np.atleast_1d(np.asarray(x))

        def _as_2d(x, width: int | None = None):
            arr = np.asarray(x)
            if arr.ndim == 1:
                if width is not None and arr.size == width:
                    return arr.reshape(1, width)
                return arr.reshape(1, -1)
            return arr

        #fetch the merger tree for the satellite
        try:
            mpb1 = self.gettree(ID, fname)
        except HTTPError as e:
            print(f'No merger tree for satellite {ID}: {e}')
            return None
        try:
            with h5.File(mpb1,'r') as f:
                #grPos = f['GroupPos'][:]
                subPos = _as_2d(f['SubhaloPos'][()], width=3)
                #grR200 = f['Group_R_Crit200'][:]
                snapnum = _as_1d(f['SnapNum'][()])
                #grVel = f['GroupVel'][:]
                subVel = _as_2d(f['SubhaloVel'][()], width=3)
                subMasstype = _as_2d(f['SubhaloMassType'][()])
                progID = _as_1d(f['SubfindID'][()])
        except Exception as e:
            print(f'Could not open/read merger tree file for satellite {ID}: {e}')
            return None
        
        #fetch the merger tree for the satellite's host

        try:
            mpbhost =self.gettree(hostID, fname)
        except HTTPError as e:
            print(f'No merger tree for host {hostID}: {e}')
            return None
        try:
            with h5.File(mpbhost,'r') as fh:
                grPos = _as_2d(fh['GroupPos'][()], width=3)
                grR200 = _as_1d(fh['Group_R_Crit200'][()])
                grVel = _as_2d(fh['SubhaloVel'][()], width=3)
                hostprog = _as_1d(fh['SubhaloID'][()])
        except Exception as e:
            print(f'Could not open/read merger tree file for host {hostID}: {e}')
            return None
        

        # to compare distances, first make arrays the same shapes
        # sometimes trees are "truncated," meaning they don't go all the way back to the start of the simulation
        if len(grPos)>len(subPos):         
            grPos=grPos[0:len(subPos),:]
            grR200=grR200[0:len(subPos)]
            grVel = grVel[0:len(subPos),:]

        elif len(grPos)<len(subPos):
            subPos=subPos[0:len(grPos),:]
            subVel=subVel[0:len(grPos),:]
            snapnum = snapnum[0:len(grPos)]
            subMasstype = subMasstype[0:len(grPos),:]
            progID = progID[0:len(grPos)]

        
        #Taking a square root is computationally inefficient.
        #Because distance in 3D is calculated with x^2+y^2+z^2 = distance^2,
        #I just work with distances in squares until the last step

        grR200sq = np.multiply(grR200,grR200) #looking within 3R200 to get joining time for more satellites
        
        gr3R200sq = 9*np.multiply(grR200,grR200) #looking within 3R200 to get joining time for more satellites
        
        #Find the relative distance between two galaxies
        difpos=np.subtract(subPos,grPos)
        
        #The simulation uses repeating boundary conditions. 
        # So two galaxies can be quite close, even if they are technically at either end of the box
        #I can explain further when we meet to discuss

        #Replace values that are affected by boundary conditions
        difpos = np.where(abs(difpos)>halfbox,abs(difpos)-L, difpos)
        
        #Find the distance square between satellite and central as a function of redshift
        distsq=np.sum(np.square(difpos),axis=1)


        wh=np.nonzero((distsq<gr3R200sq))
        
        #print(wh.any)
        #Separation at z=0
        sep_z0 = np.sqrt(distsq[0])
        sep_norm = sep_z0/grR200[0]
        #Find when the 
        closeind = np.argmin(distsq)
        closest = np.sqrt(distsq[closeind])
        closest_norm = closest/grR200[closeind]
        closest_z = self.getredshift(int(snapnum[closeind]))
        
        #Find when the satellite first crosses one,two,three R_200
        #aka find the index at which the distsq less than the correesponding R200sq
        cross1R200ind = np.nonzero((distsq<=1*grR200sq))
        cross2R200ind = np.nonzero((distsq<=4*grR200sq))
        cross3R200ind = np.nonzero((distsq<=9*grR200sq))
        if cross1R200ind is not None and len(cross1R200ind[0])>0:
            inside = snapnum[cross1R200ind]
            first1R200JoinSnap = int(inside[np.argmin(inside)])
            first1R200Redshift = self.getredshift(first1R200JoinSnap)
        else:
            first1R200Redshift = np.nan
            
        if cross2R200ind is not None and len(cross2R200ind[0])>0:
            inside = snapnum[cross2R200ind]
            first2R200JoinSnap = int(inside[np.argmin(inside)])
            first2R200Redshift = self.getredshift(first2R200JoinSnap)
        else:
            first2R200Redshift = np.nan
            
        if cross3R200ind is not None and len(cross3R200ind[0])>0:
            inside = snapnum[cross3R200ind]
            first3R200JoinSnap = int(inside[np.argmin(inside)])
            first3R200Redshift = self.getredshift(first3R200JoinSnap)
        else:
            first3R200Redshift = np.nan
        
        
        if len(wh[0])==0: 
            #in some cases, the satellite has never approached within the required distance
            #this shouldn't trigger when satellite joining redshift is defined by SubhaloGrNr
            # print('not within 3')
            return(np.nan,sep_z0,sep_norm, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, closest,closest_norm,closest_z, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        
        #find the index at which the satellite joined 
        joinind = max(wh[0])
        #print(snapnum[joinind])
        #print(grR200sq[wh],distsq[wh])
        whinside=snapnum[wh]
        whID = progID[wh] #ID of the "progenitor" galaxy of the subhalo, used to pull progenitor cutout in other code
        #print(whinside)
        minind = np.argmin(whinside)
        
        joinsnap=int(whinside[minind])
        joinprog = whID[minind]
        #print(joinsnap, snapnum, len(snapnum))
        joinred = self.getredshift(joinsnap)
        #joinind = 99-joinsnap
        #if joinind == len(snapnum): joinind=-1
        
        #Join scale factor (a), a=1 at z=0
        a=1./(1+joinred) #needed for converting comoving kiloparsecs, don't worry about this for magnetic field project
        
        #Get change in parameters since satellite joined:
        
        #Change in gas mass
        M_gas_0 = subMasstype[0][0]
        M_gas_join = subMasstype[joinind][0]
        del_M = M_gas_0 - M_gas_join
        
        #to understand kinetic energy, want to know the typical change in different masses
        del_M_total = np.sum(subMasstype[0]) - np.sum(subMasstype[joinind])
        del_M_dm = subMasstype[0][1] - subMasstype[joinind][1]
        del_M_stars = subMasstype[0][4] - subMasstype[joinind][4]
        
        #Change in kinetic energy (both all mass and star and gas mass)
        M_all_0 = np.sum(subMasstype[0])
        M_all_join = np.sum(subMasstype[joinind])
        M_lim_0 = M_gas_0 + subMasstype[0][4] #gas+stars
        M_lim_join = M_gas_join + subMasstype[joinind][4]
        
        rel_velsq_0 = np.sum(np.square((subVel[0]-grVel[0])))
        #print(rel_velsq_0)
        rel_velsq_join = np.sum(np.square((subVel[joinind]-grVel[joinind]*(1/a))))
        
        del_vsq = rel_velsq_0-rel_velsq_join
        
        #kinetic energy
        T_all_0 = 0.5*M_all_0*rel_velsq_0
        T_all_join = 0.5*M_all_join*rel_velsq_join
        del_T_all = T_all_0 - T_all_join
        
        T_lim_0 = 0.5*M_lim_0*rel_velsq_0
        T_lim_join = 0.5*M_lim_join*rel_velsq_join
        del_T_lim = T_lim_0 - T_lim_join
        
        #angular momentum
        L_0 = M_all_0*np.sqrt(rel_velsq_0)*sep_z0
        L_join = M_all_join*np.sqrt(rel_velsq_join)*np.sqrt(distsq[joinind])
        del_L = L_0-L_join
        
        # print(joinsnap)
        
        s_mass_j = subMasstype[joinind][4]
        
        hostprog_ID  = hostprog[joinind] if joinind < len(hostprog) else hostprog[-1]
        
        return(joinred,sep_z0,sep_norm, del_M, del_T_all, del_T_lim, del_M_total,del_M_dm, del_M_stars,del_vsq,joinsnap,closest, closest_norm, closest_z, joinprog, del_L, s_mass_j, hostprog_ID, L_join, L_0, first1R200Redshift, first2R200Redshift, first3R200Redshift)
