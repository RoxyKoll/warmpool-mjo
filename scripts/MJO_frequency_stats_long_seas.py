## phase calculation using PC1 and PC2 values

import numpy as np
import scipy as sc


# %matplotlib notebook

def theta1_cal( pc1, pc2 ):
    theta1           =  np.zeros(pc2.size)
    angle180         =  np.arctan2(-pc2, -pc1) * 180 / np.pi
    indNeg           =  np.where(angle180<0)[0]
    angle360         = angle180
    angle360[indNeg] = angle180[indNeg] + 360
    theta1           = np.around( ( angle360 + 22.5 ) * 8 / 360 )

    
    return theta1
 
        

#%% phase data in extended winter of each year, should provide no. of year, start year and starting index.

def theta1_winter(theta1,amp,year,ny,start_index,nod):
    the1_wintr = np.zeros((ny,), dtype=np.object)
    amp_wintr = np.zeros((ny,),dtype=np.object)
    for i in range(ny):
        if ((year+1)%4==0):
            the1_wintr[i]=theta1[start_index:start_index+nod+1]
            amp_wintr[i]=amp[start_index:start_index+nod+1]
            start_index=start_index+366
            year=year+1
        else:
            the1_wintr[i]=theta1[start_index:start_index+nod]
            amp_wintr[i]=amp[start_index:start_index+nod]
            start_index=start_index+365
            year=year+1
            
    return the1_wintr,amp_wintr


#%% Selecting the MJO segments



def select_seg(a):
    new_arr= a*1
    prev = 0
    splits = np.append(np.where(np.diff(a) != 0)[0],len(a)+1)+1
    t=0
    w=[]
    seg=[]
    for split in splits:
        #print (np.arange(0,a.size,1)[prev:split])
        w.append(list(np.arange(0,a.size,1)[prev:split]))
        prev = split
        t=t+1
        
    for i in np.arange(len(w)):
        if len(w[i])<10:
            new_arr[w[i]]=0
        elif not np.any(new_arr[w[i]]):
            continue
        else:
            seg.append(w[i])
    
    
    return new_arr,seg


#%% Finding the phase occurances for each year NOV-APR  (considerung MJO segments)

def phase_occurances(the1_obj,amp_obj,thers_amp):
    to_dur = np.zeros((len(the1_obj),8))
    mean_amp = np.zeros((len(the1_obj),8))
    for i in range(the1_obj.size):
        th=the1_obj[i]
        aa=amp_obj[i]
        a_temp      =   (aa>thers_amp)*1
        new_arr,seg =   select_seg(a_temp)
#         print(len(seg))
        for k in range(8):
            ind_f            =  np.where(th==k+1)[0]
            ind_s            =  np.where(new_arr==1)[0]
            values           =  np.intersect1d(ind_f,ind_s)
            if np.any(len(values)):
                to_dur[i,k]  =  values.size
                mean_amp[i,k]=  np.nanmean(aa[values])
            else:
                to_dur[i,k]  =  0
                mean_amp[i,k]=  0
                            
    ano_to_dur    = to_dur-np.nanmean(to_dur,axis=0)
    ano_mean_amp    = mean_amp-np.nanmean(mean_amp,axis=0)
    
        
    return    ano_to_dur,ano_mean_amp        
                     

#%% Finding the phase occurances for each year NOV-APR (considering all MJO active days)
    
def phase_occurances2(the1_obj,amp_obj,thers_amp):
    to_dur = np.zeros((the1_obj.size,8))
    mean_amp = np.zeros((the1_obj.size,8))
    for i in range(the1_obj.size):
        th=the1_obj[i]
        aa=amp_obj[i]
        for j in range(8):
            ind_f=np.where(th==j+1)[0]
            ind_s=np.where(aa>=thers_amp)[0]
            values=np.intersect1d(ind_f,ind_s)
            if np.any(len(values)):
                to_dur[i,j]  =  values.size
                mean_amp[i,j]=  np.nanmean(aa[values])
            else:
                to_dur[i,j]  =  0
                mean_amp[i,j]=  0
                            
    ano_to_dur    = to_dur-np.nanmean(to_dur,axis=0)
    ano_mean_amp    = mean_amp-np.nanmean(mean_amp,axis=0)
            
    return    to_dur,mean_amp      
    
    
    

 
