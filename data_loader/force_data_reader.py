import numpy as np
import pickle

class ForceDataReader():
    def __init__(self,pickle_file):
        self.dataset=pickle.load(open(pickle_file,'rb'))
        print ('Dataset size: ',len(self.dataset))

    def get_full_data(self):
        return self.get_batch_data(0, len(self.dataset))
    
    def get_batch_data(self, start, stop):
        electrodes_raw=[]
        electrodes_tared=[]
        out_force=[]

        for f_idx in range(start,stop):
            sample = self.dataset[f_idx]
            electrodes_raw.append(np.ravel(sample['raw_bt_electrode']))
            electrodes_tared.append(np.ravel(sample['tare_bt_electrode']))

            out_force.append(np.ravel(sample['sim_force']))

        return np.matrix(electrodes_raw), np.matrix(electrodes_tared), np.matrix(out_force)
