from myproject.utilities.iapi_TNG import get
import os

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
        snaps = get( sim['snapshots'] )

    #also pulled from join time code
    def getsub(snapnum,subid):
        #Pull url of a sub
        #subid is the ID back into the subhalo group catalog
        url= f'https://www.tng-project.org/api/{self.sim}/snapshots/'+str(snapnum)+'/subhalos/'+str(subid)+'/'
        sub=get(url)
        return(sub)

    def gettree(subid, fname:str=''):
        #pull the z=0 merger tree for a subhlao
        #some subhalos don't have merger trees
        fname = fname+'/sublink_mpb_'+str(subid) if fname != '' else ''
        fName = 'Trees/sublink_mpb_'+str(subid) if fname == '' else fname
        if os.path.exists(fName+'.hdf5'):
            return(fName+'.hdf5')
        url=f'https://www.tng-project.org/api/{self.sim}/snapshots/{self.snapshot}/subhalos/'+str(subid)+'/sublink/mpb.hdf5'
        tree=get(url,fName=fName)
        return(tree)

    def getredshift(snapnum):
        #convert a snapshot number to a redshift
        return(snaps[snapnum]['redshift'])
        
    def computeJoinTimes(ID,hostID, L, halfbox, fname:str):
        """
        Use the satellite and host trees to find the joining redshift of a satellite based on when it first approached its z=0 FoF group within 3R200
        Identify the change in satellite parameters since they joined
        """


        #fetch the merger tree for the satellite
        mpb1 = gettree(ID, fname)
        f = h5.File(mpb1,'r')
        #grPos = f['GroupPos'][:]
        subPos = f['SubhaloPos'][:]
        #grR200 = f['Group_R_Crit200'][:]
        snapnum= f['SnapNum'][:]
        #grVel = f['GroupVel'][:]
        subVel = f['SubhaloVel'][:]
        subMasstype = f['SubhaloMassType'][:]
        progID = f['SubfindID'][:]
        
        #fetch the merger tree for the satellite's host

        mpbhost =gettree(hostID, fname)
        #print(mpbhost)
        fh = h5.File(mpbhost,'r')
        grPos = fh['GroupPos'][:]
        grR200 = fh['Group_R_Crit200'][:]
        grVel = fh['SubhaloVel'][:]
        hostprog = fh['SubhaloID']
        

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

        
        #Taking a square root is computationally inefficient.
        #Because distance in 3D is calculated with x^2+y^2+z^2 = distance^2,
        #I just work with distances in squares until the last step

        grR200sq = 9*np.multiply(grR200,grR200) #looking within 3R200 to get joining time for more satellites
        
        #Find the relative distance between two galaxies
        difpos=np.subtract(subPos,grPos)
        
        #The simulation uses repeating boundary conditions. 
        # So two galaxies can be quite close, even if they are technically at either end of the box
        #I can explain further when we meet to discuss

        #Replace values that are affected by boundary conditions
        difpos = np.where(abs(difpos)>halfbox,abs(difpos)-L, difpos)
        
        #Find the distance square between satellite and central as a function of redshift
        distsq=np.sum(np.square(difpos),axis=1)


        wh=np.nonzero((distsq<grR200sq))
        
        #print(wh.any)
        #Separation at z=0
        sep_z0 = np.sqrt(distsq[0])
        sep_norm = sep_z0/grR200[0]
        #Find when the 
        closeind = np.argmin(distsq)
        closest = np.sqrt(distsq[closeind])
        closest_norm = closest/grR200[closeind]
        closest_z = getredshift(snapnum[closeind])
        
        if len(wh[0])==0: 
            #in some cases, the satellite has never approached within the required distance
            #this shouldn't trigger when satellite joining redshift is defined by SubhaloGrNr
            print('not within 3')
            return(np.nan,sep_z0,sep_norm, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, closest,closest_norm,closest_z, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        
        #find the index at which the satellite joined 
        joinind = max(wh[0])
        #print(snapnum[joinind])
        #print(grR200sq[wh],distsq[wh])
        whinside=snapnum[wh]
        whID = progID[wh] #ID of the "progenitor" galaxy of the subhalo, used to pull progenitor cutout in other code
        #print(whinside)
        minind = np.argmin(whinside)
        
        joinsnap=whinside[minind]
        joinprog = whID[minind]
        #print(joinsnap, snapnum, len(snapnum))
        joinred = getredshift(joinsnap)
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
        
        print(joinsnap)
        
        s_mass_j = subMasstype[joinind][4]
        
        hostprog_ID  = hostprog[joinind]
        
        return(joinred,sep_z0,sep_norm, del_M, del_T_all, del_T_lim, del_M_total,del_M_dm, del_M_stars,del_vsq,joinsnap,closest, closest_norm, closest_z, joinprog, del_L, s_mass_j, hostprog_ID, L_join, L_0)
