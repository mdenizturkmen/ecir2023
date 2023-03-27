from prettytable import PrettyTable
from utils import *
from ecirCodes.utils import discriminative_power

collections = ['TREC-5', 'TREC-8', 'WebTrack14', 'DeepLearning2020']
relevance_threshold = 1    # minimum relevance grade for a document to have to be considered relevant

metrics = ['map', 'P.100']

significance_test = 'hsd'  # Tukey's HSD
alphas = [0.5, 1.0]  # alpha values to be used in P_rareness and AP_rareness



dp_results = {}

for collection_no, collection in enumerate(collections):
    print(f'-----------------------------------------')
    print(f'------------ {collection} ---------------')
    print(f'-----------------------------------------')

    qrels_path = f'collections/{collection}/qrels.txt'
    inputs_dir = f'collections/{collection}/inputs'

    topk = 100

    dp_results[collection] = {}

    ########################################################################################################################################
    ####################################################      Baseline Metrics      #####################################################
    ########################################################################################################################################
    for metric in metrics:
        print(f'{metric}')
        dp_results[collection][metric] = []
        baseline_scores = trec_eval_multiple_runs(metric, qrels_path, inputs_dir, query_specific=True, relevance_threshold=relevance_threshold, topk=topk)

        significant_pairs_05 = discriminative_power(baseline_scores, alpha=0.05, test=significance_test)
        significant_pairs_01 = discriminative_power(baseline_scores, alpha=0.01, test=significance_test)
        dp_results[collection][metric].append(significant_pairs_05)
        dp_results[collection][metric].append(significant_pairs_01)
    ########################################################################################################################################
    #########################################################       P_Rareness     #########################################################
    ########################################################################################################################################
    for alpha_no, alpha in enumerate(alphas):
        print(f'P_Rareness alpha= {alphas[alpha_no]}')
        dp_results[collection][f'P_Rareness_{alpha}'] = []
        p_rareness_scores = rareness_based_trec_eval(qrels_path, inputs_dir, p_rareness_score_function, top_k=topk,
                                                         relevance_threshold=relevance_threshold, alpha=alpha)

        significant_pairs_05 = discriminative_power(p_rareness_scores, alpha=0.05, test=significance_test)
        significant_pairs_01 = discriminative_power(p_rareness_scores, alpha=0.01, test=significance_test)
        dp_results[collection][f'P_Rareness_{alpha}'].append(significant_pairs_05)
        dp_results[collection][f'P_Rareness_{alpha}'].append(significant_pairs_01)
    print('')
    ########################################################################################################################################
    ##########################################################    AP_Rareness     ##########################################################
    ########################################################################################################################################
    for alpha_no, alpha in enumerate(alphas):
        print(f'AP_Rareness alpha= {alphas[alpha_no]}')
        dp_results[collection][f'AP_Rareness_{alpha}'] = []
        ap_rareness_scores = rareness_based_trec_eval(qrels_path, inputs_dir, ap_rareness_score_function, top_k=topk, relevance_threshold=relevance_threshold, alpha=alpha)

        significant_pairs_05 = discriminative_power(p_rareness_scores, alpha=0.05, test=significance_test)
        significant_pairs_01 = discriminative_power(p_rareness_scores, alpha=0.01, test=significance_test)
        dp_results[collection][f'AP_Rareness_{alpha}'].append(significant_pairs_05)
        dp_results[collection][f'AP_Rareness_{alpha}'].append(significant_pairs_01)
    print('')


result_table = PrettyTable(['Metrics'] + [f'{c}' for c in collections])
result_table.add_row([''] + [f'95%   99%' for c in collections])
result_table.add_row([''] + ['' for c in collections])

for metric in metrics + [f'P_Rareness_{a}' for a in alphas] + [f'AP_Rareness_{a}' for a in alphas]:
    result_table.add_row([metric] + [f'{dp_results[c][metric][0]}  {dp_results[c][metric][1]}' for c in collections])

print(result_table)
