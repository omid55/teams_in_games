def compute_all_cluster_then_classify_senarios(data, metric='euclidean'):

    #import do_classification


    X = np.asmatrix(data.ix[:,:-1])
    y = np.asanyarray(data['label'])

    if metric == 'euclidean':
        linkages = ['ward', 'average', 'complete']
    else:
        linkages = ['average', 'complete']

    total_accuracy_details = []
    total_accuracy = []
    classifier_keys = []
    for number_of_clusters in range(1,10):
        if number_of_clusters == 1:
            #without clustering
            accuracy = do_classification(data, verbose=False)
            total_accuracy_details.append([len(data), accuracy])
            classifier_keys = list(accuracy.keys())
            accuracy_df = pd.DataFrame(columns=classifier_keys)
            total_accuracy.append([number_of_clusters, '-']+[accuracy[k] for k in classifier_keys])
        else:
            #with hierarchical clustering
            for linkage in linkages:
                clustering = AgglomerativeClustering(linkage=linkage, n_clusters=number_of_clusters, affinity=metric)
                clustering.fit(X)
                cluster_size_accuracy = []
                acc = 0
                for cluster_index in range(number_of_clusters):
                    idx = np.where(clustering.labels_==cluster_index)[0]
                    if len(idx) > 10:
                        accuracy = do_classification(np.column_stack((X[idx,:], y[idx])), verbose=False)
                        cluster_size_accuracy = cluster_size_accuracy + [len(idx), accuracy]
                        acc += len(idx) * np.array([accuracy[k] for k in classifier_keys])
                acc /= len(data)
                total_accuracy_details.append(cluster_size_accuracy)
                total_accuracy.append([number_of_clusters, linkage]+list(acc))
    return pd.DataFrame(data=total_accuracy, columns=['#cluster', 'linkage']+classifier_keys), total_accuracy_details