# Omid55
def plot_diversity(data, metric='euclidean'):

    if data.shape[1] == 265:
        winners = [match[:132] if not match[-1] else match[132:-1] for match in data]
        losers = [match[132:-1] if not match[-1] else match[:132] for match in data]
    else:
        winners = [match[:131] for match in data]
        losers = [match[132:] for match in data]

    winner_distance = [compute_average_distance([champions_stats_normalized[champ_id] for champ_id in np.where(winner == 1)[0]], metric=metric) for winner in winners]
    loser_distance = [compute_average_distance([champions_stats_normalized[champ_id] for champ_id in np.where(loser == 1)[0]], metric=metric) for loser in losers]

    plt.hist(winner_distance)
    plt.hist(loser_distance)
    plt.legend(['winners', 'losers'])
    plt.title(metric)
    plt.show()
    

def compute_average_distance(team, metric):
    return np.average(pdist(team, metric=metric))