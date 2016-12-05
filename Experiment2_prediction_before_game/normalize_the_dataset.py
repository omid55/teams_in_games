# Omid55
def normalize_the_dataset(dataset):
	from sklearn.preprocessing import StandardScaler

	# MIN MAX Normalization
	#x = dataset.values #returns a numpy array
	#min_max_scaler = preprocessing.MinMaxScaler()
	#x_scaled = min_max_scaler.fit_transform(x)
	#dataset = pd.DataFrame(x_scaled)

	# Standard Normalization (x-mean(x) / std(x))
	x = dataset.values[:,:-1] #returns a numpy array
	min_max_scaler = StandardScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	dataset = pd.DataFrame(np.column_stack((x_scaled,dataset.values[:,-1])), columns=dataset.columns)

	return dataset