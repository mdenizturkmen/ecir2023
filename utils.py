import pandas as pd
from os import listdir
import math
import re
import subprocess
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel
from functools import partial
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np

def extract_qrels(qrels_path):
    with open(qrels_path, 'r') as file:
        qrels = file.read()
    qrels_list = []
    for line in qrels.splitlines():
        cols = list(filter(None, re.split(' |\t', line)))
        qrels_list.append([cols[0], cols[2], int(cols[3])])
    qrels_df = pd.DataFrame(qrels_list, columns=['qid', 'docid', 'relevance'])
    topics = qrels_df.qid.unique()
    return qrels_df, topics


def relevance_set(qrels_df, topics, relevance_threshold):
    relevance_values = {}
    for q in topics:
        relevance_values[q] = set()
        relevant_docs = qrels_df[(qrels_df['qid'] == q) & (qrels_df['relevance'] >= relevance_threshold)][['docid', 'relevance']].values
        for doc in relevant_docs:
            docid, relevance = doc
            relevance_values[q].add(docid)
    return relevance_values


def document_retrieval_count(inputs_dir, topics, topk, relevance_values):
    retrieve_counts = {}

    for topic in relevance_values.keys():
        retrieve_counts[topic] = {}
        for doc in relevance_values[topic]:
            retrieve_counts[topic][doc] = 0

    for run in sorted(listdir(inputs_dir)):
        with open(f'{inputs_dir}/{run}', 'r') as file:
            input = file.read()

        input_list = []
        for line in input.splitlines():
            cols = list(filter(None, re.split(' |\t', line)))
            input_list.append([cols[0], cols[2], float(cols[4])])

        input_df = pd.DataFrame(input_list, columns=['qid', 'docid', 'score'])

        for q in topics:
            submission = input_df[input_df['qid'] == q].sort_values(by=['score', 'docid'], ascending=False).docid
            if topk != None:
                submission = submission.head(topk)
            for doc in submission:
                if relevance_values[q].__contains__(doc):
                    retrieve_counts[q][doc] += 1

    return retrieve_counts

def compute_scores(inputs_dir, topics, relevance_values, retrieve_counts, score_function, top_k, alpha):
    scores = {}
    number_of_system = len(listdir(inputs_dir))
    number_of_topic = len(topics)
    for run in sorted(listdir(inputs_dir)):
        with open(f'{inputs_dir}/{run}', 'r') as file:
            input = file.read()

        input_list = []
        for line in input.splitlines():
            cols = list(filter(None, re.split(' |\t', line)))
            input_list.append([cols[0], cols[2], float(cols[4])])
        team = run
        input_df = pd.DataFrame(input_list, columns=['qid', 'docid', 'score'])

        scores[team] = {}
        for q in topics:
            run_query_score = score_function(input_df, relevance_values, q, retrieve_counts, number_of_system, top_k, alpha)
            scores[team][q] = run_query_score
        scores[team]['all'] = sum(scores[team].values()) / number_of_topic
    return scores


def rareness_based_trec_eval(qrels_path, inputs_dir, score_function, top_k, relevance_threshold, alpha):
    qrels_df, topics = extract_qrels(qrels_path)
    relevance_values = relevance_set(qrels_df,topics, relevance_threshold)
    retrieve_counts = document_retrieval_count(inputs_dir, topics, topk=top_k, relevance_values=relevance_values)
    scores = compute_scores(inputs_dir, topics, relevance_values, retrieve_counts, score_function, top_k, alpha)
    return scores

def p_rareness_score_function(input_df, relevance_values, q, retrieve_counts, number_of_system, top_k, alpha):
    run_query_score = 0.0
    checked_alpha = alpha
    if alpha == None:
        number_of_relevant_document = len(relevance_values[q])
        if number_of_relevant_document == 0:
            return 0.0
        checked_alpha = 1 / number_of_relevant_document

    submission = input_df[input_df['qid'] == q].sort_values(by=['score', 'docid'], ascending=False).docid.head(top_k)
    for doc in submission:
        if relevance_values[q].__contains__(doc):
            h_i = 1 - (retrieve_counts[q][doc] / number_of_system)
            run_query_score += 1 + h_i * checked_alpha
    return run_query_score / top_k


def ap_rareness_score_function(input_df, relevance_values, q, retrieve_counts, number_of_system, top_k, alpha):
    number_of_relevant_document = len(relevance_values[q])
    if number_of_relevant_document == 0:
        return 0.0
    checked_alpha = alpha
    if alpha == None:
        checked_alpha = 1 / number_of_relevant_document

    submission = input_df[input_df['qid'] == q].sort_values(by=['score', 'docid'], ascending=False).docid.head(top_k)
    total_precision = 0.0
    precision_k = 0.0
    k = 1
    for doc in submission:
        if relevance_values[q].__contains__(doc):
            h_i = 1 - (retrieve_counts[q][doc] / number_of_system)
            precision_k += (1 + h_i * checked_alpha)
            total_precision += precision_k / k
        k+=1

    if number_of_relevant_document == 0:
        return 0
    return total_precision / number_of_relevant_document


def trec_eval(metric: str, qrel_path: str, input_path: str, query_specific: bool, relevance_threshold : int, topk : int):
    options = '-qm' if query_specific else '-m'
    command = ['trec_eval', options, metric, qrel_path, input_path, f'-l{relevance_threshold}', f'-M {topk}']
    result = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')

    query_score_dict = {}
    for line in result.splitlines():
        cols = line.replace(' ', '').split('\t')
        query_score_dict[cols[1]] = float(cols[2])

    return query_score_dict


def trec_eval_multiple_runs(metric: str, qrel_path: str, inputs_dir: str, query_specific: bool, relevance_threshold : int, topk : int):
    scores = {}
    for run in sorted(listdir(inputs_dir)):
        scores[run] = trec_eval(metric, qrel_path, f'{inputs_dir}/{run}', query_specific, relevance_threshold, topk)
    return scores



def discriminative_power(metric_scores, alpha, test='t-test'):
    sigtest = None
    if test == 't-test':
        sigtest = ttest_rel
    elif test == 'wilcoxon':
        sigtest = partial(wilcoxon, correction=True)
    elif test == 'hsd':
        sigtest = pairwise_tukeyhsd

    significant_pairs = 0
    t = 1
    team_names = list(metric_scores.keys())
    for i in range(len(metric_scores)):
        team1 = team_names[i]
        scores1 = list(metric_scores[team1].values())
        scores1.pop()
        for j in range(i+1,len(metric_scores)):
            print(f'%{t*200/(len(metric_scores)*(len(metric_scores)-1))}\r', end='')
            team2 = team_names[j]
            scores2 = list(metric_scores[team2].values())
            scores2.pop()
            if test == 'hsd':
                endog = scores1 + scores2
                groups = np.repeat(['a', 'b'], repeats=len(scores1))
                tukey_result = sigtest(endog=endog,groups=groups, alpha=alpha)
                p = tukey_result.pvalues[0]
            else:
                stat, p = sigtest(scores1, scores2)
            if tukey_result.reject[0]:
                significant_pairs += 1
            t+=1
    print('')
    return significant_pairs
