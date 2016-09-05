% Omid55
warning off;
% result1 = convex_optimization_problem('winner_minus_loser.csv', 0, [0.1 0.3 0.5 1 5 7 10 20 100 1000]);
% result2 = convex_optimization_problem('winner_minus_loser.csv', 1, [0.1 0.3 0.5 1 5 7 10 20 100 1000], 0.1);
result3 = convex_optimization_problem('winner_minus_loser.csv', 1, [0.1 0.3 0.5 1 5 7 10 20 100 1000], 1);
result4 = convex_optimization_problem('winner_minus_loser.csv', 1, [0.1 0.3 0.5 1 5 7 10 20 100 1000], 1000);

all_close_str = 'Positive:  89.94017283403501, Zero:  0.0, Negative:  10.05982716596499';
save('different_results1');