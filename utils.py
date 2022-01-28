import numpy as np
from Analyses_ import findseq, ETL
import json
from tqdm import tqdm
import pdb


def create_dict(mouseName, unitNr, data, hypnoState, sf, apply_artifact, removeCell, start, end):
    return {'mouseName':mouseName, 
            'unitNr':unitNr, 
            'data':data, 
            'hypnoState':hypnoState, 
            'samplingRate':sf, 
            'appliedArtifact':apply_artifact, 
            'removedCell':removeCell, 
            'startOfEpoch':start, 
            'endOfEpoch': end}

def create_list(data, hypno, mouseName, sf, apply_artifact, removeCell):
    
    # initialize list
    myList = []
    
    # looping over units
    for i in range(hypno.shape[1]):
        unit_seqs = []
        unit_seqs = findseq(hypno[:,i])

        for state, start, end, duration in zip(unit_seqs.state, unit_seqs.start_index, unit_seqs.end_index, unit_seqs.duration):
            myList.append(create_dict(mouseName=mouseName, unitNr=i, 
                                      data=data[start:end + 1, :, i], 
                                      hypnoState=state, sf = sf, 
                                      apply_artifact=apply_artifact, 
                                      removeCell=removeCell, 
                                      start = start, end = end))
            
    return myList

def get_data(address, sf):
    
    # using ETL from Analysis file
    data_load = ETL(sf = sf, ending='.mat', unit_length=600)
    data_load.get_path(address)
    data_load.load_files()

    # get data values
    data = data_load.get_data_values()
    hypno = data_load.get_hypno()
    
    print(f'data and hypno shapes in file {address} are: {data.shape}, {hypno.shape}')
    
    return data, hypno

def find_type2_locations(dictLists, pattern = [0,-2], cond1_min=100, cond1_max = 10000, cond2_min = 100, cond2_max = 10000):
    
    """
    This function takes dictionary list as input and based on given pattern it returns 
    the locations in the list where that pattern happens. # Location of first element of the pattern.
    
    dictLists: List of dictionaries
    pattern: pattern to search 0--> awake, -2-->SWS, -3-->REM
    cond1_min: min epoch duration of the first element of the pattern
    cond1_max: max epoch duration of the first element of the pattern
    cond2_min: min epoch duration of the second element of the pattern
    cond2_max: max epoch duration of the second element of the pattern
    Hint: all condition will be calculated in sample based NOT in time based
    """
    # first, reading all hypno states to a list
    all_states = [myDict['hypnoState'] for myDict in dictLists]
    
    # second, control condition on each epoch based on condition on first and second elements of the pattern.
    # first element
    window_cond1 = [(myDict['epochDuration']>cond1_min and myDict['epochDuration']<cond1_max) for myDict in dictLists]
    
    # second element
    window_cond2 = [(myDict['epochDuration']>cond2_min and myDict['epochDuration']<cond2_max) for myDict in dictLists]
    
    # reading all mouse names (needs to be checked to see if pattern comming from same mouse)
    all_names = [myDict['mouseName'] for myDict in dictLists]
    
    # finding unique mouse name
    # unique_name = np.unique(all_names)
    
    # finding the locations matching to a given pattern
    locs = np.where((np.array(all_states[:-1]) == pattern[0]) & # first element of pattern
                     (np.array(all_states[1:]) == pattern[1]) & # second element of the pattern
                     (np.array(all_names[:-1] == np.array(all_names[1:]))) & # pattern comming from same mouse
                     np.array(window_cond1[:-1]) & # condition on first element
                     np.array(window_cond2[1:]))[0] # condition on second element
    
    return locs

def find_type3_locations(dictLists, pattern = [-2,-3, -2], cond1_min=100, cond1_max = 10000, 
                         cond2_min = 100, cond2_max = 10000, cond3_min = 100, cond3_max = 10000):
    
    """
    This function takes dictionary list as input and based on given pattern it returns 
    the locations in the list where that pattern happens. # Location of first element of the pattern.
    
    dictLists: List of dictionaries
    pattern: pattern to search 0--> awake, -2-->SWS, -3-->REM
    cond1_min: min epoch duration of the first element of the pattern
    cond1_max: max epoch duration of the first element of the pattern
    cond2_min: min epoch duration of the second element of the pattern
    cond2_max: max epoch duration of the second element of the pattern
    Hint: all condition will be calculated in sample based NOT in time based
    """
    # first, reading all hypno states to a list
    all_states = [myDict['hypnoState'] for myDict in dictLists]
    
    # second, control condition on each epoch based on condition on first and second elements of the pattern.
    # first element
    window_cond1 = [(myDict['epochDuration']>cond1_min and myDict['epochDuration']<cond1_max) for myDict in dictLists]
    
    # second element
    window_cond2 = [(myDict['epochDuration']>cond2_min and myDict['epochDuration']<cond2_max) for myDict in dictLists]
    
    # third element
    window_cond3 = [(myDict['epochDuration']>cond3_min and myDict['epochDuration']<cond3_max) for myDict in dictLists]
    
    # reading all mouse names (needs to be checked to see if pattern comming from same mouse)
    all_names = [myDict['mouseName'] for myDict in dictLists]
    
    # finding unique mouse name
    # unique_name = np.unique(all_names)
    
    # finding the locations matching to a given pattern
    locs = np.where((np.array(all_states[:-1]) == pattern[0]) & # first element of pattern
                     (np.array(all_states[1:]) == pattern[1]) & # second element of the pattern
                     (np.array(all_names[:-1] == np.array(all_names[1:]))) & # pattern comming from same mouse
                     np.array(window_cond1[:-1]) & # condition on first element
                     np.array(window_cond2[1:]))[0] # condition on second element
    
    # getting all locations matching 3 pattern
    temp_loc3 = []
    temp_loc3 = np.where(np.array(all_states) == pattern[2])[0]
    
    out = []
    
    # searching for right candidates
    for i, can in enumerate(locs + 1):
        
        temp = []
        temp = np.where(temp_loc3 > can)[0] # all locations in pattern 3
        #pdb.set_trace()
        if temp.any() and all_names[temp_loc3[temp[0]]] == all_names[can] and window_cond3[temp_loc3[temp[0]]]:
            out.append([can - 1, can, temp_loc3[temp[0]]])
    
    return np.vstack(out)


def find_type4_locations(dictLists, pattern = [0, -2,-3, -2], cond1_min=100, cond1_max = 10000, 
                         cond2_min = 100, cond2_max = 10000, cond3_min = 100, cond3_max = 10000, 
                         cond4_min = 100, cond4_max = 10000):
    
    """
    This function takes dictionary list as input and based on given pattern it returns 
    the locations in the list where that pattern happens. # Location of first element of the pattern.
    
    dictLists: List of dictionaries
    pattern: pattern to search 0--> awake, -2-->SWS, -3-->REM
    condx_min: min epoch duration of the x element of the pattern
    condx_max: max epoch duration of the x element of the pattern
    
    Hint: all condition will be calculated in sample based NOT in time based
    """
    # first, reading all hypno states to a list
    all_states = [myDict['hypnoState'] for myDict in dictLists]
    
    # second, control condition on each epoch based on condition on first and second elements of the pattern.
    # first element
    window_cond1 = [(myDict['epochDuration']>cond1_min and myDict['epochDuration']<cond1_max) for myDict in dictLists]
    
    # find last three element's patter using type3
    type3_locs = find_type3_locations(dictLists=dictLists, pattern=pattern[1:], 
                                      cond1_min = cond2_min, cond1_max = cond2_max, 
                                      cond2_min = cond3_min, cond2_max = cond3_max, 
                                      cond3_min = cond4_min, cond3_max = cond4_max)
    
    # reading all mouse names (needs to be checked to see if pattern comming from same mouse)
    all_names = [myDict['mouseName'] for myDict in dictLists]
    
    
    # getting all locations matching 1 pattern
    temp_loc1 = np.array([i for i in range(len(dictLists)) if dictLists[i]['hypnoState'] == 0])
    
    out = []
    
    # searching for right candidates
    for i in range(type3_locs.shape[0]):
        if i==0:
            temp = []
            temp = temp_loc1[np.where(temp_loc1<type3_locs[i,0])[0]]
        else:
            temp = []
            temp = temp_loc1[np.where((temp_loc1<type3_locs[i,0]) & (temp_loc1>type3_locs[i-1,2]))[0]]
        #pdb.set_trace()
        if temp.any() and window_cond1[temp[-1]] and (all_names[temp[-1]] == all_names[type3_locs[i,0]] == all_names[type3_locs[i,1]] == all_names[type3_locs[i,2]]):
            out.append(np.hstack([temp[-1], type3_locs[i,:]]))
    
    return np.vstack(out)

def applyArtifacts(data, artifacts):
    if artifacts:
        for artifact in artifacts:
            data[artifact[0]:artifact[1], :, artifact[2]] = 0
        
        return data
    return data

def removeCells(data, cells):
    if cells:
        return np.delete(data, cells, axis = 1)
    return data

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def avg_firing(data, sf):
    # return in avgerage in second
    avg = np.nansum(np.where(data>0 , 1, np.nan), axis = 0)/(data.shape[0]/sf)
    return np.where(avg>0, avg, np.nan)

def avg_amplitude(data):
    # return in avgerage amplitude
    return np.nanmean(np.where(data>0 , data, np.nan), axis = 0)

def nrActiveCells(data):
    return len(data[~np.isnan(data)])
    
def div_avg_firing(data, sf, n =3):
    spaces = np.linspace(0, data.shape[0], n+1, dtype = int)
    avg = [avg_firing(data=data[spaces[i]:spaces[i+1],:], sf=sf) for i in range(n)]
    return avg
    
def div_avg_amplitude(data, n =3):
    spaces = np.linspace(0, data.shape[0], n+1, dtype = int)
    avg = [avg_amplitude(data=data[spaces[i]:spaces[i+1],:]) for i in range(n)]
    return avg

def top_least(data, prc = 10, top = True):
    
    """
    This function return indices and values of top or least x% of given data
    
    data: array
    prc: percentage
    top: if True returns top if false returns least
    """
    data = np.array(data)
    
    indx = np.argsort(data)
    
    notNans = ~np.isnan(data[indx])
    
    selc = max(1, int(len(data) * (prc/100)))
    
    if top:
        return indx[notNans][-selc:], data[indx[notNans]][-selc:]
    
    if not top:
        return indx[notNans][:selc], data[indx[notNans]][:selc]
    
    
    
    
def cellID(indices, mouse_name, all_mice_names):
    
    '''
    This function return explicit cell ID based on list of indices and mouse name
    '''
    
    all_mice_names = np.unique(all_mice_names)
    
    Dict = {}
    for i, name in enumerate(all_mice_names):
        Dict.update({name: (i+1) * 1000})
    
    return  indices + Dict[mouse_name]


def spindle_locs(data, mouse_nr = 0, unit_nr = 0, unit_length = 18597, min_duration = 0):
    
    '''
    This function takes spindle information from slo_spi function as list 
    and return location of spindles per given mouse and unit name/number.
    
    ** input data shows indices (start to the end) where spindles are not happening
       and each unit indices are starting from zero
    
    data: slo_spi output per mouse as list
    mouse_nr: int to select mouse
    unit_nr: int to select unit within mouse
    unit_length: length of unit in sample
    min_duration: minimum window duration for selected epoch
    
    '''
    
    # taking mouse specific part of data and making numpy array
    data = np.vstack(data[mouse_nr])
    
    # selecting given unit from data
    data_selected = np.squeeze(data[data[:,2] == unit_nr,:])
    
    # check if selected file in not empty
    if not data_selected.any():
        raise Exception('selected data is empty!!!')
    
    # create vecotr for locations
    target_locs = np.ones(unit_length, dtype = np.int16)
    
    if data_selected.ndim == 1:
        
        target_locs[data_selected[0]: data_selected[1]] = 0
        
    else:
        
        for i in range(data_selected.shape[0]):
            
            target_locs[data_selected[i, 0]: data_selected[i,1]] = 0
           
        
    # getting location start and end
    Seq = findseq(target_locs)
    
    # cleaning zeros out
    Seq = Seq[Seq.state == 1]
    
    # minimum duration
    if Seq.to_numpy().any():
        Seq = Seq[Seq.duration > min_duration]
    
    return target_locs, Seq



def search_spindle(data_spindle, Dict, mouse_index):
    """
    This function is searching for spindles in a given Dictionary from data_list
    
    data_spindle: spindle data (spi_slo_art_all)
    Dict: individual dictionary from data_list
    mouse_index: Dictionary of mouse names and corresponding indices
    """
    # initialize output
    data=[]; selected_locations=[]; not_indices = []
    
    temp = spindle_locs(data = data_spindle, mouse_nr=mouse_index[Dict['mouseName']], 
                        unit_nr= Dict['unitNr'], min_duration=15)[1].to_numpy()
    if temp.any():
        locations = np.squeeze(np.where((temp[:,1] >= Dict['startOfEpoch']) & (temp[:,2]<=Dict['endOfEpoch'])))
        
        if locations.any():
            
            selected_locations = temp[locations]
            
            data = []
            if selected_locations.ndim == 1:
                data = Dict['data'][(selected_locations[1] - Dict['startOfEpoch']) : (selected_locations[1] - Dict['startOfEpoch']) + selected_locations[3], :]
                not_indices.append(np.arange(selected_locations[1] - Dict['startOfEpoch'], selected_locations[1] - Dict['startOfEpoch'] + selected_locations[3] - 1))
            else:
                for i in range(selected_locations.shape[0]):
                    data.append(Dict['data'][(selected_locations[i, 1] - Dict['startOfEpoch']) : (selected_locations[i, 1] - Dict['startOfEpoch']) + selected_locations[i, 3], :])
                    not_indices.append(np.arange(selected_locations[i, 1] - Dict['startOfEpoch'], selected_locations[i, 1] - Dict['startOfEpoch'] + selected_locations[i, 3] - 1))
             
            not_indices = np.hstack(not_indices)
            
    return data, selected_locations, np.delete(Dict['data'], not_indices, axis = 0), not_indices


def spi_inOut_diff(x, y):
    
    """
    This function calculates the difference between avg_within spindle epoch to avg_outside spindle epoch
    
    x: within spindle/oscillation average
    y: outside spindle/oscillation average
    """
    
    # first chaning nan to 0
    x = np.where(np.isnan(x), 0, x)
    y = np.where(np.isnan(y), 0, y)
    
    # calculate difference
    diff = x - y
    
    # change 0 to nan (when cell within and outside spindle epoch doesn't show any activity)
    diff = np.where(diff == 0, np.nan, diff)
    
    return diff


def zscore_calculator(dataBase, mouse_name_list, variable = "frequency", divided_data = False):
    """
    This function calculates zscore per cell in a given mouse. 
    dataBase --> Full data base to add zscore key to each dictionary
    mouse_name_list --> list of all mouse names avaible in database
    variable --> must be one of "frequency" or "amplitude"
    """

    if variable == "frequency" and not divided_data:
        # zscore normalization of avg firing rate. Normalization is applied per cell for each individual mouse separately
        mouse_zscore_info = []
        for mouse in tqdm(mouse_name_list):
            mouse_avg = []
            for dict in dataBase:
                if dict["mouseName"] == mouse:
                    mouse_avg.append(dict["avg_firing_perSecond"])
            mouse_zscore_info.append({"mouseName":mouse, "nanmean": np.nanmean(np.stack(mouse_avg), axis = 0), "nanstd": np.nanstd(np.stack(mouse_avg), axis = 0)})
            

        # update database
        for dict in tqdm(dataBase):
            for info in mouse_zscore_info:
                if dict["mouseName"] == info["mouseName"]:
                    dict.update({"avg_firing_perSecond_zscore": (dict["avg_firing_perSecond"] - info["nanmean"]) / info["nanstd"]})
        
        return dataBase

    elif variable == "frequency" and divided_data:
        # zscore normalization of avg firing rate. Normalization is applied per cell for each individual mouse separately
        mouse_zscore_info = []
        for mouse in tqdm(mouse_name_list):
            mouse_avg = []
            for dict in dataBase:
                if dict["mouseName"] == mouse:
                    mouse_avg.append(dict["div_avg_firing_perSecond"])
            mouse_zscore_info.append({"mouseName":mouse, "nanmean": np.nanmean(np.vstack(mouse_avg), axis = 0), "nanstd": np.nanstd(np.vstack(mouse_avg), axis = 0)})
            

        # update database
        for dict in tqdm(dataBase):
            for info in mouse_zscore_info:
                if dict["mouseName"] == info["mouseName"]:
                    dict.update({"div_avg_firing_perSecond_zscore": [(el - info["nanmean"]) / info["nanstd"] for el in dict["div_avg_firing_perSecond"]]})

        return dataBase
    
    elif variable == "amplitude" and not divided_data:
        # zscore normalization of avg firing rate. Normalization is applied per cell for each individual mouse separately
        mouse_zscore_info = []
        for mouse in tqdm(mouse_name_list):
            mouse_avg = []
            for dict in dataBase:
                if dict["mouseName"] == mouse:
                    mouse_avg.append(dict["avg_amplitude"])
            mouse_zscore_info.append({"mouseName":mouse, "nanmean": np.nanmean(np.stack(mouse_avg), axis = 0), "nanstd": np.nanstd(np.stack(mouse_avg), axis = 0)})
            

        # update database
        for dict in tqdm(dataBase):
            for info in mouse_zscore_info:
                if dict["mouseName"] == info["mouseName"]:
                    dict.update({"avg_amplitude_zscore": (dict["avg_amplitude"] - info["nanmean"]) / info["nanstd"]})

        return dataBase

    elif variable == "amplitude" and divided_data:
        mouse_zscore_info = []
        for mouse in tqdm(mouse_name_list):
            mouse_avg = []
            for dict in dataBase:
                if dict["mouseName"] == mouse:
                    mouse_avg.append(dict["div_avg_amplitude"])
            mouse_zscore_info.append({"mouseName":mouse, "nanmean": np.nanmean(np.vstack(mouse_avg), axis = 0), "nanstd": np.nanstd(np.vstack(mouse_avg), axis = 0)})
            

        # update database
        for dict in tqdm(dataBase):
            for info in mouse_zscore_info:
                if dict["mouseName"] == info["mouseName"]:
                    dict.update({"div_avg_amplitude_zscore": [(el - info["nanmean"]) / info["nanstd"] for el in dict["div_avg_amplitude"]]})

        return dataBase

    else:
        raise Exception("Input variable might be wrong!")


def raster_data(dataBase, mouse_name):
    """
    This function takes full data base and mouse name then returns concatinated events for a given mouse (raster data)
    dataBase --> full data base
    mouse_name --> name of the mouse from data base
    """

    rasters = []
    hypno = []
    for dict in dataBase:
        if dict["mouseName"] == mouse_name:
            rasters.append(np.where(dict["data"]>0, 1, 0))
            hypno.append(np.zeros((dict["data"].shape[0])) + dict["hypnoState"])
    
    return np.vstack(rasters), np.hstack(hypno)