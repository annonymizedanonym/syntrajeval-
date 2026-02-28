"""Convert encoded csv files to one-hot-encoded npy files."""

import pandas as pd
import numpy as np
import argparse

def data_conversion(df, tid_col,test=False):
    """Converts input panda dataframe to one-hot-encoded Numpy array (locations are still in float)."""
    
    x = [[] for i in ['lat_lon', 'day', 'hour', 'category', 'mask','tid','count']]
    for tid in df[tid_col].unique():
        traj = df.loc[df[tid_col].isin([tid])]
        features = np.transpose(traj.loc[:, ['lat', 'lon', 'day', 'hour', 'category']].values)
        loc_list = []
        for i in range(0, len(traj)):
            lat = traj['lat'].values[i]
            lon = traj['lon'].values[i]
            loc_list.append(np.array([lat, lon], dtype=np.float64))
        x[0].append(loc_list)
        x[1].append(np.eye(7)[features[2].astype(np.int32)])
        x[2].append(np.eye(24)[features[3].astype(np.int32)])
        x[3].append(np.eye(10)[features[4].astype(np.int32)])
        x[4].append(np.ones(shape=(features[0].shape[0],1))) # mask
        
        if test:
            x[5].append(tid)
            x[6].append(traj.tid.count())
        

    converted_data = np.array([np.array(f, dtype=object) for f in x])
    return converted_data

def data_deconversion(x_test, x_prediction, df_test, max_length):
    """
    Converts one-hot-encoded Numpy array prediction (locations are still in float) to pd dataframe.

    x_test: original test set
    x_prediction: analog synthetic set
    
    """
    traj_attr_concat_list = []
    for attributes in x_prediction:
        traj_attr_list = []
        idx = 0
        for row in attributes:
            if row.shape == (max_length, 2):
                traj_attr_list.append(row[max_length-x_test[6][idx][0]:])
            else:
                traj_attr_list.append(np.argmax(row[max_length-x_test[6][idx][0]:],axis=1).reshape(x_test[6][idx][0],1))
            idx += 1
        traj_attr_concat = np.concatenate(traj_attr_list)
        traj_attr_concat_list.append(traj_attr_concat)
    traj_data = np.concatenate(traj_attr_concat_list,axis=1)

    label = np.array(df_test['label']).reshape(-1,1)
    tid = np.array(df_test['tid']).reshape(-1,1)
    traj_data = np.concatenate([label,tid,traj_data],axis=1)
    df_traj_fin = pd.DataFrame(traj_data)
        
    df_traj_fin.columns = ['label','tid','lat','lon','day', 'hour', 'category','mask']    
        
    del df_traj_fin['mask']
        
    df_traj_fin['tid'] = df_traj_fin['tid'].astype(np.int32)
    df_traj_fin['day'] = df_traj_fin['day'].astype(np.int32)
    df_traj_fin['hour'] = df_traj_fin['hour'].astype(np.int32)
    df_traj_fin['category'] = df_traj_fin['category'].astype(np.int32)
    df_traj_fin['label'] = df_traj_fin['label'].astype(np.int32)

    return df_traj_fin

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, default="dev_train_encoded_final.csv")
    parser.add_argument("--save_path", type=str, default="train_encoded.npy")
    parser.add_argument("--tid_col", type=str, default="tid")
    args = parser.parse_args()
    
    df = pd.read_csv(args.load_path)
    converted_data = data_conversion(df, args.tid_col)
    np.save(args.save_path, converted_data)