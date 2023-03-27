import copy
from utils import *
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def get_ordered_topic_retrieval_counts(qrels_df, retrieve_counts, topic, increasing: bool):
    relevant_documents = qrels_df[(qrels_df['qid'] == topic) & (qrels_df['relevance'] == 1)]
    topic_retrieval_counts = {}
    for doc in relevant_documents.docid:
        if not retrieve_counts[topic].keys().__contains__(doc):
            topic_retrieval_counts[doc] = 0
        else:
            topic_retrieval_counts[doc] = retrieve_counts[topic][doc]

    sorted_topic_retrieval_counts = {k: v for k, v in sorted(topic_retrieval_counts.items(), key=lambda item: item[1],
                                                             reverse=(not increasing))}
    return [run for run, score in sorted_topic_retrieval_counts.items()], [score for run, score in
                                                                           sorted_topic_retrieval_counts.items()]


def per_topic_rank(scores, teams):
    scores_teams = zip(scores, teams)
    scores_teams.sort()
    sorted_scores = [scr for scr, tm in scores_teams]
    sorted_teams = [tm for scr, tm in scores_teams]
    return sorted_teams, sorted_scores


def relevant_document_retrieval_count_topk(inputs_dir, item_dict, topics, topk):
    retrieve_counts = {}
    for run in sorted(listdir(inputs_dir)):
        with open(f'{inputs_dir}/{run}', 'r') as file:
            input = file.read()

        input_list = []
        for line in input.splitlines():
            cols = list(filter(None, re.split(' |\t', line)))
            input_list.append([cols[0], cols[2], int(cols[3])])
        input_df = pd.DataFrame(input_list, columns=['qid', 'docid', 'retrieval_order'])

        for q in topics:
            if not retrieve_counts.keys().__contains__(q):
                retrieve_counts[q] = {}
            submission = input_df[input_df['qid'] == q].sort_values('retrieval_order').docid.head(topk)
            for doc in submission:
                if not item_dict[q].keys().__contains__(doc):
                    continue
                if retrieve_counts[q].keys().__contains__(doc):
                    retrieve_counts[q][doc] += 1
                else:
                    retrieve_counts[q][doc] = 1
    return retrieve_counts


# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
def get_retrieval_matrix(inputs_dir, relevance_values, topic, topk):
    number_of_systems = len(listdir(inputs_dir))
    number_of_relevant_document = len(relevance_values[topic])
    retrieval_matrix = np.zeros(shape=(number_of_relevant_document, number_of_systems), dtype='int')

    document_dict = {}
    reverse_document_dict = {}
    doc_idx = 0
    for rel_doc in sorted(relevance_values[topic]):
        document_dict[rel_doc] = doc_idx
        reverse_document_dict[doc_idx] = rel_doc
        doc_idx += 1

    run_dict = {}
    run_idx = 0
    for run in sorted(listdir(inputs_dir)):
        run_dict[run] = run_idx
        with open(f'{inputs_dir}/{run}', 'r') as file:
            input = file.read()

        input_list = []
        for line in input.splitlines():
            cols = list(filter(None, re.split(' |\t', line)))
            if cols[0] == topic:
                input_list.append([cols[0], cols[2], float(cols[4])])
        input_df = pd.DataFrame(input_list, columns=['qid', 'docid', 'score'])

        submission = input_df[input_df['qid'] == topic].sort_values(by=['score', 'docid'], ascending=False).docid.head(
            topk)
        for doc in submission:
            if relevance_values[topic].__contains__(doc):
                document_index = document_dict[doc]
                retrieval_matrix[document_index, run_idx] = 1
        run_idx += 1
    return retrieval_matrix, document_dict, run_dict, reverse_document_dict


def get_precision(inputs_dir, relevance_values, topic, topk):
    precision_list = []
    for run in sorted(listdir(inputs_dir)):
        with open(f'{inputs_dir}/{run}', 'r') as file:
            input = file.read()

        input_list = []
        for line in input.splitlines():
            cols = list(filter(None, re.split(' |\t', line)))
            if cols[0] == topic:
                input_list.append([cols[0], cols[2], float(cols[4])])
        input_df = pd.DataFrame(input_list, columns=['qid', 'docid', 'score'])

        submission = input_df[input_df['qid'] == topic].sort_values(by=['score', 'docid'], ascending=False).docid.head(
            topk)
        relevant_ret = 0
        for doc in submission:
            if relevance_values[topic].__contains__(doc):
                relevant_ret += 1

        precision = relevant_ret / topk
        precision = int(round(round(precision, 4) * topk))
        precision_list.append(precision)

    return sorted(precision_list, reverse=True)


def calculate_scores(retrieval_matrix, topic_retrieval_counts, reverse_document_dict, topk, alpha):
    doc_num, number_of_systems = retrieval_matrix.shape
    score_list = []

    for team_no in range(number_of_systems):
        team_score = 0.0
        for doc_no in range(doc_num):
            if retrieval_matrix[doc_no, team_no] == 0:
                continue

            doc_name = reverse_document_dict[doc_no]
            doc_retrieve_count = topic_retrieval_counts[doc_name]

            h_i = 1 - (doc_retrieve_count / (number_of_systems + 1))
            team_score += 1 + alpha * h_i

        team_score /= topk
        score_list.append(team_score)

    return score_list


# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------


collection = 'DeepLearning2020'
relevance_threshold = 1

qrels_path = f'collections/{collection}/qrels.txt'
inputs_dir = f'collections/{collection}/inputs'

# Precision_rareness
score_function = p_rareness_score_function

topk = 100            # cutoff threshold
alphas = [0, 0.5, 1]  # alpha values to be used in P_rareness

qrels_df, topics = extract_qrels(qrels_path)
relevance_values = relevance_set(qrels_df, topics, relevance_threshold)
retrieve_counts = document_retrieval_count(inputs_dir, topics, topk=topk, relevance_values=relevance_values)

retrieve_steps = [i + 1 for i in range(45)]  # number of steps for the experiment  (at each step, the synthetic system retrieves one more relevant document)
topic = topics[38]                           # topic to use in the experiment

########################################################################################################################
# S_rare
########################################################################################################################


results = np.zeros(shape=(len(retrieve_steps), len(alphas)), dtype='float')   # result matrix

for alpha_idx in range(len(alphas)):  # for each alpha value
    alpha = alphas[alpha_idx]
    print(f'alpha:{alpha}')

    # compute scores of original systems
    hardness_based_scores = rareness_based_trec_eval(qrels_path, inputs_dir, score_function, top_k=topk, relevance_threshold=relevance_threshold, alpha=alpha)
    topic_scores = [hardness_based_scores[team][topic] for team in hardness_based_scores.keys()]
    sorted_topic_scores = sorted(topic_scores, reverse=False)
    number_of_systems = len(hardness_based_scores)
    max_h_i = 1 - (1 / number_of_systems)

    # at each step, increase the relevant retrieval of the synthetic system
    for retrieve_num in retrieve_steps:

        # score of the synthetic system
        synthetic_system_score = retrieve_num * (1 + max_h_i * alpha) / 100

        # gather scores of original systems and the synthetic system
        extended_sorted_topic_scores = sorted(sorted_topic_scores + [synthetic_system_score], reverse=False)

        # find the ranking of the synthetic system
        equal_ranks = []
        idx = 1
        for score in extended_sorted_topic_scores:
            if score == synthetic_system_score:
                equal_ranks.append(idx)
            elif synthetic_system_score < score:
                break
            idx += 1

        if len(equal_ranks) == 0:
            equal_ranks.append(idx)
        avg_idx = np.average(equal_ranks)
        results[retrieve_num - 1, alpha_idx] = number_of_systems - avg_idx + 2


colors = cm.rainbow(np.linspace(0, 1, len(alphas)))
markers = ['o', 's', 'X', '^']

x = retrieve_steps

for alpha_no, alpha in enumerate(alphas):
    y = results[:,alpha_no]
    plt.plot(x, y, color=colors[alpha_no], marker=markers[alpha_no], label=f'P_Rareness_{alpha}')
plt.legend()
plt.show()

#############################################################################################################3
# S_common
#############################################################################################################3

retrieval_matrix, document_dict, run_dict, reverse_document_dict = get_retrieval_matrix(inputs_dir, relevance_values, topic, topk)

results = np.zeros(shape=(len(retrieve_steps), len(alphas)), dtype='float')   # result matrix

for alpha_idx in range(len(alphas)):   # for each alpha value
    alpha = alphas[alpha_idx]
    print(f'alpha:{alpha}')

    # sort documents from most retrieved to least retrieved
    topic_retrieval_counts = copy.deepcopy(retrieve_counts[topic])
    sorted_topic_retrieval_counts = {k: v for k, v in sorted(topic_retrieval_counts.items(), key=lambda item: item[1], reverse=True)}
    ordered_topic_docs, ordered_topic_retrieval_counts = [run for run, score in sorted_topic_retrieval_counts.items()], [score for run, score in sorted_topic_retrieval_counts.items()]
    number_of_systems = len(listdir(inputs_dir))


    synthetic_system_score = 0.0

    # at each step, increase the relevant retrieval of the synthetic system
    for retrieve_num in retrieve_steps:

        if retrieve_num > len(ordered_topic_docs):
            break

        # get the most commonly retrieved document
        retrieved_doc = ordered_topic_docs[retrieve_num - 1]
        retrieved_doc_retrieve_num = ordered_topic_retrieval_counts[retrieve_num - 1]
        doc_idx = document_dict[retrieved_doc]
        topic_retrieval_counts[retrieved_doc] += 1

        # rareness bonus for the synthetic system
        new_h_i = 1 - ((retrieved_doc_retrieve_num + 1) / (number_of_systems + 1))
        synthetic_system_score += (1 + new_h_i * alpha)

        # compute scores of original systems
        topic_scores = calculate_scores(retrieval_matrix, topic_retrieval_counts, reverse_document_dict, topk, alpha)

        # score of the synthetic system

        my_score = synthetic_system_score / topk

        extended_sorted_topic_scores = sorted(list(topic_scores) + [my_score], reverse=False)

        # find the ranking of the synthetic system
        equal_ranks = []
        idx = 1
        for score in extended_sorted_topic_scores:
            if score == my_score:
                equal_ranks.append(idx)
            elif my_score < score:
                break
            idx += 1

        if len(equal_ranks) == 0:
            equal_ranks.append(idx)
        avg_idx = np.average(equal_ranks)
        results[retrieve_num - 1, alpha_idx] = number_of_systems - avg_idx + 2




colors = cm.rainbow(np.linspace(0, 1, len(alphas)))
markers = ['o', 's', 'X', '^']

x = retrieve_steps

for alpha_no, alpha in enumerate(alphas):
    y = results[:,alpha_no]
    plt.plot(x, y, color=colors[alpha_no], marker=markers[alpha_no], label=f'P_Rareness_{alpha}')
plt.legend()
plt.show()