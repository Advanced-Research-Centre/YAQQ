from qiskit.extensions import UnitaryGate
import numpy as np
from datetime import datetime

ds = []
expname = 'results/data/u-bmw/'
u = np.load(expname+'first_adder.npy')
ds.append(UnitaryGate(u,label='3qU'))
u = np.load(expname+'check_condition.npy')
ds.append(UnitaryGate(u,label='3qU'))
u = np.load(expname+'IQFT.npy')
ds.append(UnitaryGate(u,label='3qU'))
u = np.load(expname+'last_adder.npy')
ds.append(UnitaryGate(u,label='3qU'))
u = np.load(expname+'lower_half_adder.npy')
ds.append(UnitaryGate(u,label='3qU'))
u = np.load(expname+'max_0B.npy')
ds.append(UnitaryGate(u,label='3qU'))
u = np.load(expname+'smallestIQFT.npy')
ds.append(UnitaryGate(u,label='3qU'))
u = np.load(expname+'upper_half_adder.npy')
ds.append(UnitaryGate(u,label='3qU'))

now = datetime.now()
exp_id = 'NUSA_eid-0010_'+now.strftime("%Y-%m-%d-%H-%M")
np.save('results/data/'+exp_id+'ds', ds)