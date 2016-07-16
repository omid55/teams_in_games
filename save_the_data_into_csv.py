#Omid55
import numpy as np
import utils
matches_reduced = utils.load_data('matches_reduced')

MAXIMUM_TEAM_SIZE = 5
# first team is the loser and second one is the winner
matches = -1 * np.ones(
    (len(matches_reduced.values()),
     MAXIMUM_TEAM_SIZE*2)
    )
members = np.zeros(
    (len(matches_reduced.values()),
     MAXIMUM_TEAM_SIZE*2)
    )

for match_index, match in enumerate(matches_reduced.values()):
    if not match.values():
        print 'Match', match_index, 'does not have any members.'
        raise
    loser_index = 0
    winner_index = MAXIMUM_TEAM_SIZE
    for member in match.values()[0]:
        if not member or 'winner' not in member or 'champLevel' not in member:
            print 'Match', match_index, 'does not have the expected structure.'
            raise
        if not member['winner']:
            if loser_index >= MAXIMUM_TEAM_SIZE:
                print 'Match', match_index, 'has more losers than expected.'
                raise
            matches[match_index][loser_index] = member['champLevel']
            members[match_index][loser_index] = member['summonerId']
            loser_index += 1
        else:
            if winner_index >= 2*MAXIMUM_TEAM_SIZE:
                print 'Match', match_index, 'has more winners than expected.'
                raise
            matches[match_index][winner_index] = member['champLevel']
            members[match_index][winner_index] = member['summonerId']
            winner_index += 1
    #matches[match_index][:MAXIMUM_TEAM_SIZE] = sorted(matches[match_index][:MAXIMUM_TEAM_SIZE], reverse=True)
    #matches[match_index][MAXIMUM_TEAM_SIZE:] = sorted(matches[match_index][MAXIMUM_TEAM_SIZE:], reverse=True)

    idx = [i[0] for i in sorted(enumerate(matches[match_index][:MAXIMUM_TEAM_SIZE]), key=lambda x: x[1], reverse=True)]
    matches[match_index][:MAXIMUM_TEAM_SIZE] = matches[match_index][idx]
    members[match_index][:MAXIMUM_TEAM_SIZE] = members[match_index][idx]
    idx = [i[0] for i in sorted(enumerate(matches[match_index][MAXIMUM_TEAM_SIZE:]), key=lambda x: x[1], reverse=True)]
    idx = [x+MAXIMUM_TEAM_SIZE for x in idx]
    matches[match_index][MAXIMUM_TEAM_SIZE:] = matches[match_index][idx]
    members[match_index][MAXIMUM_TEAM_SIZE:] = members[match_index][idx]

np.savetxt(
    'matches_reduced.csv',
    matches,
    fmt='%d',
    delimiter=',',
    newline='\n',  # new line character
    footer='',  # file footer
    comments='',  # character to use for comments
    header='loser1,loser2,loser3,loser4,loser5,winner1,winner2,winner3,winner4,winner5'  # file header
    )

np.savetxt(
    'members_in_matches_reduced.csv',
    members,
    fmt='%d',
    delimiter=',',
    newline='\n',  # new line character
    footer='',  # file footer
    comments='',  # character to use for comments
    header='loser1,loser2,loser3,loser4,loser5,winner1,winner2,winner3,winner4,winner5'  # file header
    )