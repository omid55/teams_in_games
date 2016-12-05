# Omid55
def print_class_ratio(data):
	assert 'label' in data, 'The dataset does not have required label column.'
	if type(data) is pd.core.frame.DataFrame:
		labels = set(data['label'])
	print('Data has', len(data), 'samples,')
	for l in labels:
		print(int(l),': ', 100 * len(np.where(data['label'] == l)[0]) / len(data), '%')