import random
from utils import *


def stability(metric_scores, topics, topic_subset_size, iteration, fuzziness):
    comparison_dict = {}
    for i in range(len(metric_scores)):
        comparison_dict[i] = {}
        for j in range(i + 1, len(metric_scores)):
            comparison_dict[i][j] = [0,0,0]

    for it in range(iteration):
        teams = list(metric_scores.keys())
        team_indexes = list(range(len(teams)))
        topic_subset1 = random.sample(population=list(topics), k=topic_subset_size)

        scores1 = topic_subset_scores(metric_scores, topic_subset1)
        team_ordering = [x for _, x in sorted(zip(scores1, team_indexes), reverse=True)]

        for i in range(len(team_ordering)):
            team1 = team_ordering[i]
            for j in range(i + 1, len(team_ordering)):
                team2 = team_ordering[j]
                min_team = min(team1, team2)
                max_team = max(team1, team2)

                team1_score = scores1[team1]
                team2_score = scores1[team2]
                significance = (team1_score - team2_score) > fuzziness
                if significance:
                    inc_idx = int(min_team == team2)
                    comparison_dict[min_team][max_team][inc_idx] += 1
                else:
                    comparison_dict[min_team][max_team][2] += 1

    total_comparison = 0
    majority_number = 0
    minority_number = 0
    tie_number = 0
    for i in range(len(metric_scores)):
        for j in range(i + 1, len(metric_scores)):
            ternary_comp = comparison_dict[i][j]
            majority = max(ternary_comp[0:2])
            minority = min(ternary_comp[0:2])
            tie = ternary_comp[2]

            majority_number += majority
            minority_number += minority
            tie_number += tie
            total_comparison += majority + minority + tie

    reliability_score = majority_number / total_comparison
    return round(reliability_score, 3)



def topic_subset_scores(metric_scores, topic_list):
    scores = []
    for team in metric_scores.keys():
        team_score = 0.0
        for topic in topic_list:
            team_score += metric_scores[team][topic]
        team_score /= len(topic_list)
        scores.append(team_score)
    return scores


collections = ['TREC-5', 'TREC-8', 'WebTrack14', 'DeepLearning2020']
relevance_threshold = 1 # minimum relevance grade for a document to have to be considered relevant

topk = 100  # cutoff threshold
alphas = [0.5, 1.0] # alpha values to be used in P_rareness and AP_rareness
fuzziness = 0.05  # fuzziness value for stability measurement
iteration = 1000    # number of iteration


baseline_metrics = ['P.100', 'map']

collection_results = {}

for collection in collections:

    print(f'------------ {collection} ---------------')
    print(f'-----------------------------------------')
    collection_results[collection] = {}

    qrels_path = f'collections/{collection}/qrels.txt'
    inputs_dir = f'collections/{collection}/inputs'

    qrels_df, topics = extract_qrels(qrels_path)
    topic_subset_size = round(len(topics) / 2)
    ##############################################################################################################################################################
    for alpha_no, alpha in enumerate(alphas):
        print(f'P_Rareness alpha= {alphas[0:alpha_no+1]}\r', end='')
        set_hardness_based_scores = rareness_based_trec_eval(qrels_path, inputs_dir, p_rareness_score_function(), top_k=topk, relevance_threshold=relevance_threshold, alpha=alpha)
        collection_results[collection][f'P_Rareness_{alpha}'] = stability(set_hardness_based_scores, topics, topic_subset_size, iteration, fuzziness)
    print('')
    ##############################################################################################################################################################
    for alpha_no, alpha in enumerate(alphas):
        print(f'AP_Rareness alpha= {alphas[0:alpha_no+1]}\r', end='')
        rank_hardness_based_scores = rareness_based_trec_eval(qrels_path, inputs_dir, ap_rareness_score_function, top_k=topk, relevance_threshold=relevance_threshold, alpha=alpha)
        collection_results[collection][f'AP_Rareness_{alpha}'] = stability(rank_hardness_based_scores, topics, topic_subset_size, iteration, fuzziness)
    print('')
    ##############################################################################################################################################################
    for metric in baseline_metrics:
        print(f'{metric}')
        baseline_scores = trec_eval_multiple_runs(metric, qrels_path, inputs_dir, query_specific=True, relevance_threshold=relevance_threshold, topk=topk)
        collection_results[collection][metric] = stability(baseline_scores, topics, topic_subset_size, iteration, fuzziness)


from prettytable import PrettyTable
result_table = PrettyTable(['Metrics'] + collections)

result_table.add_row([f'Precision@100'] + [collection_results[c]['P.100'] for c in collections])
result_table.add_row([f'P_Rareness_0.5'] + [collection_results[c]['P_Rareness_0.5'] for c in collections])
result_table.add_row([f'P_Rareness_1.0'] + [collection_results[c]['P_Rareness_1.0'] for c in collections])

result_table.add_row([f'AP'] + [collection_results[c]['map'] for c in collections])
result_table.add_row([f'AP_Rareness_0.5'] + [collection_results[c]['AP_Rareness_0.5'] for c in collections])
result_table.add_row([f'AP_Rareness_1.0'] + [collection_results[c]['AP_Rareness_1.0'] for c in collections])

print(result_table)


