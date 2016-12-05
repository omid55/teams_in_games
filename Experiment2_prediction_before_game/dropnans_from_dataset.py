# Omid55
def dropnans_from_dataset(dataset):
	to_be_deleted = []
	for idx,item in enumerate(dataset.as_matrix()):
	    if np.isnan(item).any():
	        to_be_deleted.append(idx)
	dataset = dataset.drop(to_be_deleted)
	return dataset