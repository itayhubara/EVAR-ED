import pandas as pd
import numpy as np

def exctract_features(data):

    Flag= 'all_data'
    num_patients=data.shape[0]
    samples_ind_all=range(num_patients)
    if Flag =='all_data':
        ind=[]
        for i in range(data.shape[1]):
            remove_flag=False
            not_nan_ind=[]
            for j in range(num_patients):
                if isinstance(data[j,i], str):
                    remove_flag=True
                    break
                elif not np.isnan(data[j,i]):
                    not_nan_ind.append(j)
            #import pdb; pdb.set_trace()
            if len(not_nan_ind)<0.90*num_patients:
                remove_flag=True
            elif not remove_flag and len(not_nan_ind)<num_patients:
                mean_value=data[not_nan_ind,i].mean()
                nan_ind=np.setdiff1d(samples_ind_all,not_nan_ind)
                data[nan_ind,i] = mean_value
            if not remove_flag:
                ind.append(i)
    return ind


data_train=pd.read_stata('./2017May12EVARPOD0AnalyticCohortDay0vsLong.dta')
features_names_all=data_train.keys()
data_train=data_train.values
data_val=pd.read_stata('2017May12EVARPOD0ValidationCohortDay0vsLong.dta')
data_val=data_val.values


label_key=[(i,key) for i, key in enumerate(features_names_all) if 'loscategory2' in key][0][0]

ind_train=exctract_features(data_train)
ind_val=exctract_features(data_val)

ind_both=np.intersect1d(ind_train,ind_val)
#import pdb; pdb.set_trace()
features_names_in=['centerid','age','dc_status','transfer','preop_dialysis','preop_creat','stress','livingstatus','hemo','hxfamaaa','preop_ef','preop_maxaaadia','unfitoaaa'
                ,'anesthesia','totalproctime','convtoopen','contrast','crystal','ebl','intraop_prbc','icustay','postop_respiratory','BMI','race3','egfr','egfrcategory','anydm'
                ,'anysmoke','anycad','anyAAA','anychf','anycopd','primary_insurer2','bypassstent','major_amp2','graftconfig','completionendoleak','postopvaso','N_centervolyear','hxpvd']
features_names_all=features_names_all[ind_both]
ind_in=[]
for n,i in  zip(features_names_all,ind_both):
    #import pdb; pdb.set_trace()
    if n in features_names_in:
        ind_in.append(i)

target_train=data_train[:,label_key]>1
target_val=data_val[:,label_key]>1
import pdb; pdb.set_trace()
target_train=target_train.astype('float32')
target_val=target_val.astype('float32')

data_train=data_train[:,ind_in]
mean_train=data_train.mean(0)
data_train=data_train-mean_train
max_train=data_train.max(0)
data_train/=max_train

data_val=data_val[:,ind_in]
data_val=data_val-mean_train
data_val/=max_train

print(len(ind_in))

#ind = np.arange(num_patients)
#np.random.shuffle(ind)

data2=dict()
#import pdb; pdb.set_trace()
data2['features_train']=data_train.astype('float32')
data2['target_train']=target_train.astype('float32')
data2['features_val']=data_val.astype('float32')
data2['target_val']=target_val.astype('float32')
np.savez('data_evar',features_train=data2['features_train'], target_train=data2['target_train'],features_val=data2['features_val'], target_val=data2['target_val'])
