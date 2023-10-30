import pandas as pd

d= {'Acc': [], 'Abc': [], 'Acb': [], 'Abb': [], 'v_ctx': [], 'v_bg': [], 'seq': [], 'att': []}
df = pd.DataFrame(data=d)
df.to_hdf('/work/jp464/striatum-sequence/output/retrieval_speed.h5', 'data')
