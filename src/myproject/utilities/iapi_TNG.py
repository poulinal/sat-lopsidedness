"""
Contains useful funcitons for fetching data with the IllustrisTNG API
For the illustrisTNG workshop at STScI Symposium April 2024
Modified by Bryanne McDonough from materials provided by the TNG team
"""


import requests
import numpy as np
import h5py
import os.path
import time
import os
from dotenv import load_dotenv
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from urllib.parse import urlparse, urljoin
import glob

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("API_KEY")

baseUrl = 'https://www.tng-project.org/api/'
headers = {"api-key" : api_key}

TIMEOUT = 600
MAX_RETRIES = 5
BACKOFF_FACTOR = 10  # seconds

def tng_get(url, stream=False):
    """GET request with manual redirect handling for TNG API."""
    from urllib.parse import urlparse, urljoin
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=headers, allow_redirects=False, timeout=TIMEOUT)

            # Handle redirects
            while r.status_code in (301, 302, 303, 307, 308):
                redirect_url = r.headers["Location"]
                if not urlparse(redirect_url).scheme:
                    redirect_url = urljoin(url, redirect_url)

                # Strip api-key for data server redirects (token is in URL)
                hostname = urlparse(redirect_url).hostname or ""
                if "data" in hostname:
                    r = requests.get(redirect_url, timeout=TIMEOUT, stream=stream)
                else:
                    r = requests.get(redirect_url, headers=headers,
                                     timeout=TIMEOUT, stream=stream)

            r.raise_for_status()
            return r

        except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            status = getattr(e.response, "status_code", None) if hasattr(e, "response") else None
            if status in (502, 503, 504) or isinstance(e, requests.exceptions.Timeout):
                wait = BACKOFF_FACTOR * (2 ** attempt)
                print(f"  Attempt {attempt+1}/{MAX_RETRIES} failed ({e}). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {url}")

def get(path, params=None, fName='temp'): # gets data from url, saves to file
    """
    Routine to pull data from online
    Credit to TNG team
    """
    # print(f"Fetching data from {path} with params {params} and saving to {fName}")

    # r = requests.get('https://www.tng-project.org/api/', headers=headers)
    # print(r.status_code)  # should be 200

    # make HTTP GET request to path
    if (len(headers['api-key'])!=32):
        print("Check your api key")
    r = requests.get(path, params=params, headers=headers, timeout=50000)
    # print(f"Response code: {r.status_code}")
    # r.raise_for_status()

    # r = requests.get(path, params=params, headers=headers, allow_redirects=False, timeout=1020)
    # # Handle redirects manually
    # while r.status_code in (301, 302, 303, 307, 308):
    #     from urllib.parse import urlparse, urljoin
    #     redirect_url = r.headers['Location']
    #     if not urlparse(redirect_url).scheme:
    #         redirect_url = urljoin(path, redirect_url)

    #     # Only strip api-key header if redirected to a data server (token is in URL)
    #     if 'data' in urlparse(redirect_url).hostname:
    #         redirect_url = redirect_url.replace('data-us.tng-project.org', 'data-eu.tng-project.org')
    #         r = requests.get(redirect_url, timeout=300)
    #     else:
    #         r = requests.get(redirect_url, headers=headers, timeout=120)  # keep headers

    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        response = r.json()
        # print(f"JSON response: {response}")  # <-- add this
        return response
        # return r.json() # parse json responses automatically

    # print(f"Saving data to {fName}")
    dataFile=fName+'.hdf5'
    # Saves to file, currently disabled
    # print(r.headers)
    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        with open(dataFile, 'wb') as f:
            f.write(r.content)
        return dataFile # return the filename string

    # print(f"Saving data to {dataFile} where r: {r}")
    return r


def getsub(snapnum,subid):
    #Pull fields associated with a given subhalo at a given snapshot
    url= 'https://www.tng-project.org/api/TNG100-1/snapshots/'+str(snapnum)+'/subhalos/'+str(subid)+'/'
    sub=get(url)
    return(sub)

def gettree(snapnum,subid):
    snapnum=str(snapnum)
    fName = 'trees/sublink_mpb_'+str(subid)
    if os.path.exists(fName+'.hdf5'):
        return(fName+'.hdf5')
    url='https://www.tng-project.org/api/TNG100-1/snapshots/'+snapnum+'/subhalos/'+str(subid)+'/sublink/mpb.hdf5'
    tree=get(url,fName=fName)
    return(tree)

def getredshift(snapnum, simname):
    
    r=get(baseUrl)
    names = [sim['name'] for sim in r['simulations']]
    # i = names.index('TNG100-1')
    i = names.index(simname)
    sim = get( r['simulations'][i]['url'] )
    
    snaps = get( sim['snapshots'] )
    
    try: 
        zs=[snaps[j]['redshift'] for j in snapnum]
        return(zs)
    except: return(snaps[snapnum]['redshift'])







def getSubhaloField(field, simulation='TNG100-1', snapshot=99,
                    fileName='tempCat', rewriteFile=0, saveFile:bool=False):
    """
    Credit to TNG team
    Data from one field for all subhalos in a given snapshot      
    
    These two commands are near identical, so I'm going to detail them both here. 
    They have the same input and output, except one deals with halos (roughly 
    speaking 'groups/ clusters') and the other with the subhalos in those halos 
    (the 'galaxies' in those 'groups'). See Intro to the Data (or Naming 
    Conventions) for more on the data structure/ naming conventions used.
    
    
    Parameters
    ----------
    field : str
        Name of the one field to be returned. The fields can be found in 
        section 2. of this page
        http://www.illustris-project.org/data/docs/specifications/

    simulation : str
        Which simulation to pull data from

    snapshot : int
        Which snapshot (moment in time) to pull data from

    The following two parameters are discussed in more detail here!

    fileName : str
        Default is 'tempGal.hdf5'. Filename for where to store or load the data 

    rewriteFile : int
        [0 or 1] If this is equal to 0 then the program will try and pull data 
        from the file specified by fileName rather than re-downloading. This can 
        save time, especially for galaxies which are large or you will work on 
        frequently, but you will only be able to access fields you originally 
        requested
        
        
    Returns
    -------
    data : array
        A numpy array containing the data for a specific field for all halos/subhalos

        
    Examples
    --------
    Let's pull out the velocity dispersion of stars in every subhalo and their 
    DM mass, and then restrict ourselves to only looking at the primary subhalo 
    in each halo (i.e. the most massive galaxy in each group).

    The velocity dispersion (N_sub values)
    
        >>> galaxyVelDisp=iApi.getSubhaloField('SubhaloVelDisp')

    The mass of each different particle type in a galaxy (N_sub x 6 values, 
    see getGalaxyData for more info on particle types)
    
        >>> galaxyMassType=iApi.getSubhaloField('SubhaloMassType') 

    The subhalo number of the primary subhalo in each halo (N_halo values)
        
        >>> primarySubhalos=iApi.getHaloField('GroupFirstSub') 

    Velocity dispersion of primary subhalos
    
        >>> velDisp=galaxyVelDisp[primarySubhalos]

    Total dark matter mass of primary subhalos 
    
        >>> mDM=galaxyMassType[primarySubhalos,1] 
    
    """

    dataFile=fileName+'.hdf5'

    print(f"doesn't exist: {not os.path.exists(dataFile)} or {rewriteFile==1}, datafile: {dataFile}")
    if not os.path.exists(dataFile) or rewriteFile==1:
        url='https://www.tng-project.org/api/'+simulation+'/files/groupcat-'+str(snapshot)+'/?Subhalo='+field

        try:
            print("tring to access api field")
            dataFile=get(url,fName=fileName)

        except:
            print("instead accessing groupcat")
            try:
                data = extract_field(chunk_dir, snapshot, group='Subhalo', field=field)
            except:
                datacatalogFolder = os.path.dirname(os.path.dirname(dataFile))
                print(f"need to download all chunks first to {datacatalogFolder}")
                if not os.path.exists(datacatalogFolder):
                    os.makedirs(datacatalogFolder)
                    print(f'created directory: {datacatalogFolder}')
                chunk_dir = download_all_chunks(simulation, snapshot, datacatalogFolder)
                print(f"\nAll chunks saved to: {chunk_dir}")
                data = extract_field(chunk_dir, snapshot, group='Subhalo', field=field)
                print(f"retrieved field")

                if saveFile:
                    savepath = fileName
                    save_field(data, field, savepath)
                return data
   
    with h5py.File(dataFile,'r') as f:
                data=np.array(f['Subhalo'][field])
        

    return data
    
  
def getHaloField(field, simulation='TNG100-1', snapshot=99,
                 fileName='tempCat', rewriteFile=0, saveFile:bool=False):
    """
    Credit to TNG team
    Data from one field for all halos/subhalos in a given snapshot      
    
    These two commands are near identical, so I'm going to detail them both here. 
    They have the same input and output, except one deals with halos (roughly 
    speaking 'groups/ clusters') and the other with the subhalos in those halos 
    (the 'galaxies' in those 'groups'). See Intro to the Data (or Naming 
    Conventions) for more on the data structure/ naming conventions used.
    
    
    Parameters
    ----------
    field : str
        Name of the one field to be returned. The fields can be found in 
        section 2. of this page
        http://www.illustris-project.org/data/docs/specifications/

    simulation : str
        Which simulation to pull data from

    snapshot : int
        Which snapshot (moment in time) to pull data from

    The following two parameters are discussed in more detail here!

    fileName : str
        Default is 'tempGal.hdf5'. Filename for where to store or load the data 

    rewriteFile : int
        [0 or 1] If this is equal to 0 then the program will try and pull data 
        from the file specified by fileName rather than re-downloading. This can 
        save time, especially for galaxies which are large or you will work on 
        frequently, but you will only be able to access fields you originally 
        requested
        
        
    Returns
    -------
    data : array
        A numpy array containing the data for a specific field for all halos/subhalos

        
    Examples
    --------
    Let's pull out the velocity dispersion of stars in every subhalo and their 
    DM mass, and then restrict ourselves to only looking at the primary subhalo 
    in each halo (i.e. the most massive galaxy in each group).

    The velocity dispersion (N_sub values)
    
        >>> galaxyVelDisp=iApi.getSubhaloField('SubhaloVelDisp')

    The mass of each different particle type in a galaxy (N_sub x 6 values, 
    see getGalaxyData for more info on particle types)
    
        >>> galaxyMassType=iApi.getSubhaloField('SubhaloMassType') 

    The subhalo number of the primary subhalo in each halo (N_halo values)
        
        >>> primarySubhalos=iApi.getHaloField('GroupFirstSub') 

    Velocity dispersion of primary subhalos
    
        >>> velDisp=galaxyVelDisp[primarySubhalos]

    Total dark matter mass of primary subhalos 
    
        >>> mDM=galaxyMassType[primarySubhalos,1] 
    
    """
    dataFile=fileName+'.hdf5'
    
    if not os.path.exists(dataFile) or rewriteFile==1:
        url='http://www.tng-project.org/api/'+simulation+'/files/groupcat-'+str(snapshot)+'/?Group='+field
        # dataFile=get(url,fName=fileName)
        try:
            print("tring to access api field")
            dataFile=get(url,fName=fileName)

        except:
            print("instead accessing groupcat")
            try:
                data = extract_field(chunk_dir, snapshot, group='Group', field=field)
            except:
                print("need to download all chunks first")
                datacatalogFolder = os.path.dirname(os.path.dirname(dataFile))
                chunk_dir = download_all_chunks(simulation, snapshot, datacatalogFolder)
                print(f"\nAll chunks saved to: {chunk_dir}")
                data = extract_field(chunk_dir, snapshot, group='Group', field=field)
                print(f"retrieved field")

                if saveFile:
                    savepath = fileName
                    save_field(data, field, savepath)
                return data

        
    with h5py.File(dataFile,'r') as f:
        data=np.array(f['Group'][field])

    return data

    
def getSubcutout(subID, parttype, params, sim='TNG100-1', snapnum='99', fName='temp'):
    """
    Obtain particle level information (params) for all particles of parttype bound to the subhalo identified by subID
    subID (int): index into subhalo catalog
    parttype (str): can be 'gas', 'dm', 'stars', 'bhs', or 'tracers'
    params (str): parameters in the PartType fields to pull, formatted as: 'param1, param2, param3, ..., paramN'
    sim (str): simulation to pull the cutout from
    snapnum (int or str): snap number to pull cutout from
    fName (str): file name to save cutout to
    """
    
    if fName!='temp' and os.path.exists(fName):
        return(fName)
    
    snapnum=str(snapnum)
    part_param = {parttype : params}
    
    sub_url = "http://www.tng-project.org/api/"+sim+"/snapshots/"+snapnum+"/subhalos/"+str(subID)+"/"
    sub=get(sub_url)
    print(sub_url)
    cutouturl=sub['cutouts']['subhalo']
    
    
    cutout = get(cutouturl,params=part_param, fName=fName)
    
    return(cutout)



def list_groupcat_chunks(sim, snap):
    """List all groupcat chunk files for a simulation/snapshot."""
    url = f"{baseUrl}{sim}/files/groupcat-{snap}/"
    print(f"Listing groupcat chunks: {url}")
    r = tng_get(url)
    data = r.json()
    # The response contains a list of file URLs
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "files" in data:
        return data["files"]
    else:
        # Try to find chunk URLs from the response
        print(f"Unexpected response format, keys: {data.keys() if isinstance(data, dict) else type(data)}")
        print(f"Response: {data}")
        return data


def download_chunk(chunk_url, outpath):
    """Download a single groupcat chunk to disk."""
    if os.path.exists(outpath):
        print(f"  Already exists: {outpath}, skipping")
        return outpath

    print(f"  Downloading: {chunk_url}")
    r = tng_get(chunk_url, stream=True)

    with open(outpath, "wb") as f:
        for block in r.iter_content(chunk_size=1024 * 1024):
            f.write(block)

    print(f"  Saved: {outpath} ({os.path.getsize(outpath) / 1e6:.1f} MB)")
    return outpath


def download_all_chunks(sim, snap, outdir, rewrite=False):
    """Download all groupcat chunks for a simulation/snapshot."""
    chunk_dir = os.path.join(outdir, f"{sim}_groupcat_{snap}")
    if rewrite and os.path.exists(chunk_dir):
        print(f"Removing existing directory: {chunk_dir}")
        import shutil
        shutil.rmtree(chunk_dir)
    elif os.path.exists(chunk_dir):
        print(f"Using existing directory: {chunk_dir}")
    else:
        os.makedirs(chunk_dir, exist_ok=False)

    chunks_info = list_groupcat_chunks(sim, snap)

    # Determine chunk count and URLs
    chunk_urls = []
    if isinstance(chunks_info, list):
        # List of dicts with 'url' keys, or list of URLs
        for item in chunks_info:
            if isinstance(item, dict) and "url" in item:
                chunk_urls.append(item["url"])
            elif isinstance(item, str):
                chunk_urls.append(item)
    elif isinstance(chunks_info, dict):
        # Try numbered approach
        n = chunks_info.get("num_files_groupcat", 0)
        for i in range(n):
            chunk_urls.append(f"{baseUrl}{sim}/files/groupcat-{snap}.{i}.hdf5")

    if not chunk_urls:
        # Fallback: try incrementing until we get a 404
        print("Could not parse chunk list, trying sequential download...")
        i = 0
        while True:
            url = f"{baseUrl}{sim}/files/groupcat-{snap}.{i}.hdf5"
            try:
                outpath = os.path.join(chunk_dir, f"groupcat_{snap}.{i}.hdf5")
                download_chunk(url, outpath)
                chunk_urls.append(url)
                i += 1
            except Exception as e:
                print(f"  Stopped at chunk {i}: {e}")
                break

        if i == 0:
            raise RuntimeError("Could not download any chunks")
        return chunk_dir

    print(f"Found {len(chunk_urls)} chunks to download")
    for i, url in enumerate(chunk_urls):
        if not rewrite and os.path.exists(os.path.join(chunk_dir, f"groupcat_{snap}.{i}.hdf5")):
            print(f"  Already exists: {chunk_dir}/groupcat_{snap}.{i}.hdf5, skipping")
            continue
        # Ensure URL has scheme
        if url.startswith("http://"):
            url = url.replace("http://", "https://")
        outpath = os.path.join(chunk_dir, f"groupcat_{snap}.{i}.hdf5")
        download_chunk(url, outpath)

    return chunk_dir


def extract_field(chunk_dir, snap, group="Subhalo", field="SubhaloFlag"):
    """Concatenate a field from all downloaded groupcat chunks."""
    
    pattern = os.path.join(chunk_dir, f"groupcat_{snap}.*.hdf5")
    files = sorted(glob.glob(pattern),
                   key=lambda x: int(x.split(".")[-2]))  # sort by chunk number

    if not files:
        raise FileNotFoundError(f"No groupcat files found: {pattern}")

    print(f"Reading '{group}/{field}' from {len(files)} chunks...")
    arrays = []
    for f in files:
        with h5py.File(f, "r") as hf:
            if group in hf and field in hf[group]:
                arrays.append(hf[group][field][:])
            else:
                available = list(hf[group].keys()) if group in hf else list(hf.keys())
                print(f"  Warning: '{group}/{field}' not in {f}")
                print(f"  Available: {available[:10]}...")

    if not arrays:
        raise KeyError(f"Field '{group}/{field}' not found in any chunk")

    data = np.concatenate(arrays)
    print(f"Concatenated {field}: shape={data.shape}, dtype={data.dtype}")
    return data


def save_field(data, field, outpath):
    """Save extracted field to HDF5."""
    with h5py.File(outpath, "w") as f:
        f.create_dataset(field, data=data)
    print(f"Saved {field} to {outpath} ({os.path.getsize(outpath) / 1e6:.1f} MB)")