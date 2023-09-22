import sys

sys.path.append('.')

from glob import glob
import configs.config_loader as cfg_loader
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import os
cfg = cfg_loader.get_config()
print(cfg.data_dir)
print(cfg.input_data_glob)

print('Finding raw files for preprocessing.')
paths = glob( "./"+cfg.data_dir + cfg.input_data_glob)
#print(paths)
paths = sorted(paths)

chunks = np.array_split(paths,cfg.num_chunks)
paths = chunks[cfg.current_chunk]


if cfg.num_cpus == -1:
	num_cpus = mp.cpu_count()
else:
	num_cpus = cfg.num_cpus

def multiprocess(func):
	p = Pool(num_cpus)
	p.map(func, paths)
	p.close()
	p.join()

print('Start deleting.')
#multiprocess(to_off)

#print('Start speed sampling.')
for path in paths:
	file_path = os.path.dirname(path)
    
	file_name = os.path.basename(path)
	for clean_up in glob(file_path+'/*.*'):
		print(clean_up)
		if not clean_up.endswith(file_name):  
			os.remove(clean_up)

	

