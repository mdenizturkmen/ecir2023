from utils import *
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

collections = ['TREC-5', 'TREC-8', 'WebTrack14', 'DeepLearning2020']
relevance_threshold = 1   # minimum relevance grade for a document to have to be considered relevant

alphas = [0, 0.25, 0.5, 0.75, 1]  # alpha values for P_rareness and AP_raraness
topk = 100  # cutoff threshold


#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

tau_dict = {}


for collection in collections:
    print(f'-----------------------------------------')
    print(f'------------ {collection} ---------------')
    print(f'-----------------------------------------')

    tau_dict[collection] = {}
    precision_taus = []
    ap_taus = []

    qrels_path = f'collections/{collection}/qrels.txt'
    inputs_dir = f'collections/{collection}/inputs'

    # -------------------------------------------------------------------------------------------------------------------
    baseline_metric_scores = trec_eval_multiple_runs(f'P.{topk}', qrels_path, inputs_dir, query_specific=True,
                                                        relevance_threshold=relevance_threshold, topk=topk)
    precision_scores = [baseline_metric_scores[team]['all'] for team in baseline_metric_scores.keys()]
    # -------------------------------------------------------------------------------------------------------------------
    for alpha_no, alpha in enumerate(alphas):
        print(f'P_Rareness alpha= {alphas[0:alpha_no+1]}\r', end='')
        score_function = p_rareness_score_function
        p_rareness_scores = rareness_based_trec_eval(qrels_path, inputs_dir, score_function, top_k=topk, relevance_threshold=relevance_threshold, alpha=alpha)
        p_rareness_scores = [p_rareness_scores[team]['all'] for team in p_rareness_scores.keys()]
        tau, _ = stats.kendalltau(p_rareness_scores, precision_scores)
        precision_taus.append(tau)
    print('')

    #-------------------------------------------------------------------------------------------------------------------
    baseline_metric_scores = trec_eval_multiple_runs('map', qrels_path, inputs_dir, query_specific=True,
                                                        relevance_threshold=relevance_threshold, topk=topk)
    map_scores = [baseline_metric_scores[team]['all'] for team in baseline_metric_scores.keys()]
    #-------------------------------------------------------------------------------------------------------------------
    for alpha_no, alpha in enumerate(alphas):
        print(f'AP_Rareness alpha= {alphas[0:alpha_no + 1]}\r', end='')
        score_function = ap_rareness_score_function
        ap_rareness_scores = rareness_based_trec_eval(qrels_path, inputs_dir, score_function, top_k=topk, relevance_threshold=relevance_threshold, alpha=alpha)
        ap_rareness_scores = [ap_rareness_scores[team]['all'] for team in ap_rareness_scores.keys()]
        tau, _ = stats.kendalltau(ap_rareness_scores, map_scores)
        ap_taus.append(tau)
    print('')
    tau_dict[collection]['P_Rareness'] = precision_taus
    tau_dict[collection]['AP_Rareness'] = ap_taus



#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

colors = cm.rainbow(np.linspace(0, 1, 2))
markers = ['o', 's', 'X', '^']


for collection in collections:
    x = alphas

    y = tau_dict[collection]['P_Rareness']
    plt.xticks(x)
    plt.plot(x, y, color=colors[0], marker=markers[0], label='P x P_Rareness')

    y = tau_dict[collection]['AP_Rareness']
    plt.xticks(x)
    plt.plot(x, y, color=colors[1], marker=markers[1], label='AP x AP_Rareness')

    plt.axis((0, 1, 0.3, 1))
    plt.show()

