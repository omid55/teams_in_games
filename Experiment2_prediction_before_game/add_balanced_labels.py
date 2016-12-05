# Omid55
'''
Adding label and swapping 50% of winners and losers; Thus:
label 0 == winner + loser
label 1 == loser + winner
'''
def add_balanced_labels(data, one_ratio=0.5):
    data_copied = pd.DataFrame(data).copy()
    data_copied['label'] = np.zeros([len(data_copied),1])
    dt = data_copied.as_matrix()
    idx = np.random.choice(len(dt), int(len(dt)*one_ratio), replace=False)
    tf = math.floor(dt.shape[1]/2)
    tmp = dt[idx,tf:2*tf]
    dt[idx,tf:2*tf] = dt[idx,:tf]
    dt[idx,:tf] = tmp
    dt[idx,-1] = 1
    #result = pd.DataFrame(data=dt, columns=data.columns)
    return data_copied