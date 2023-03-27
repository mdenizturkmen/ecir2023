import random
import scipy.stats as stats
import matplotlib.pyplot as plt
from utils import *


def compute_scores_for_sampled_participant(inputs_dir, topics, relevance_values, retrieve_counts, score_function, top_k, alpha, participant_list):
    '''
    :param inputs_dir: directory with runs' submission
    :param topics: topic list of the collection used
    :param relevance_values: relevance status of the documents, e.g., qrels
    :param retrieve_counts: dictionary keeping which document is retrieved by how many systems
    :param score_function: Determines which metric of ours is used (Precision_rareness or AP_rareness)
    :param top_k: cutoff threshold
    :param alpha: alpha value for our metrics
    :param participant_list: list of participants included in the scoring
    :return: dictionary -> scores of participants included in the scoring
    '''
    scores = {}
    number_of_system = len(participant_list)
    number_of_topic = len(topics)
    for run in sorted(listdir(inputs_dir)):
        if not participant_list.__contains__(run):
            continue

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


def document_retrieval_per_participant(inputs_dir, topics, topk, relevance_values):
    '''
    :param inputs_dir: directory with runs' submission
    :param topics: topic list of the collection used
    :param topk: cutoff threshold
    :param relevance_values: relevance status of the documents, e.g., qrels
    :return: dictionary -> which team retrieves which documents for each topic
    '''
    retrieve_counts = {}
    team_list = sorted(listdir(inputs_dir))

    for team in team_list:
        retrieve_counts[team] = {}
        for topic in topics:
            retrieve_counts[team][topic] = set()

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
                    retrieve_counts[run][q].add(doc)
    return retrieve_counts


def extract_retrieve_counts_for_participants(document_retrieval_per_participant, participant_list, topics):
    '''
    :param document_retrieval_per_participant: which team retrieves which documents for each topic
    :param participant_list: list of participants to be considered
    :param topics: topic list of the collection used
    :return:
    '''
    filtered_retrieve_counts = {}
    for topic in relevance_values.keys():
        filtered_retrieve_counts[topic] = {}
        for doc in relevance_values[topic]:
            filtered_retrieve_counts[topic][doc] = 0

    for team in participant_list:
        for topic in topics:
            for doc in document_retrieval_per_participant[team][topic]:
                filtered_retrieve_counts[topic][doc] += 1
    return filtered_retrieve_counts

collections = ['TREC-5', 'TREC-8', 'Robust2004', 'WebTrack14', 'DeepLearning2020']

collection_no = 1
collection = collections[collection_no]
relevance_threshold = 1 # minimum relevance grade for a document to have to be considered relevant

qrels_path = f'collections/{collection}/qrels.txt'
inputs_dir = f'collections/{collection}/inputs'

# main score calculation (when all teams are included in scoring)
# -------------------------------------------------------------------------------------------------------
qrels_df, topics = extract_qrels(qrels_path)
relevance_values = relevance_set(qrels_df, topics, relevance_threshold)

# either
score_function = p_rareness_score_function    # method: "Precision_rareness"
# or
score_function = ap_rareness_score_function   # method: "AP_rareness"

topk = 100    # cutoff threshold
alpha = 1.0   # alpha parameter for our metrics

# main scorings (when all teams are included in scoring)
main_scores = rareness_based_trec_eval(qrels_path, inputs_dir, score_function, top_k=topk, relevance_threshold=relevance_threshold, alpha=alpha)

# -------------------------------------------------------------------------------------------------------


# which team retrieves which documents for each topic
document_retrieval_dict = document_retrieval_per_participant(inputs_dir, topics, topk=topk, relevance_values=relevance_values)



iteration = 1000  # number of iteration
iteration = 2  # number of iteration
team_counts = [2, 4, 8, 16, 32, 64]  # how many teams to sample in each iteration
team_counts = [2]  # how many teams to sample in each iteration

avg_tau_per_team_count = []

for number_of_team_to_sample in team_counts:
    taus = []
    for i in range(iteration):
        team_list = sorted(listdir(inputs_dir))
        sampled_teams = sorted(random.sample(population=team_list, k=number_of_team_to_sample))  # sample teams

        # retrieval counts of documents based on sampled participant list
        retrieve_counts_for_sampled_participants = extract_retrieve_counts_for_participants(document_retrieval_dict, sampled_teams, topics)

        # scores of sampled participants
        participant_scores = compute_scores_for_sampled_participant(inputs_dir, topics, relevance_values, retrieve_counts_for_sampled_participants, score_function, topk, alpha, participant_list=sampled_teams)

        main_scores_list = []
        sampled_scores = []
        for participant in sampled_teams:
            main_scores_list.append(main_scores[participant]['all'])
            sampled_scores.append(participant_scores[participant]['all'])

        tau, _ = stats.kendalltau(main_scores_list, sampled_scores)
        taus.append(tau)
        print(f'iteration {i} -> tau:{tau}')

    avg_tau = np.mean(taus)
    avg_tau_per_team_count.append(avg_tau)



# plot tau correlation w.r.t. number of participants
ax1_label = f"# of participants"
ax2_label = "average tau correlation"
plot_title = f"#of participant impact"

fig, ax = plt.subplots(figsize=(10, 8), facecolor=plt.cm.Blues(.2), sharex=True)
plt.scatter(team_counts, avg_tau_per_team_count, c='salmon', s=150)
plt.plot(team_counts, avg_tau_per_team_count, c='red')
plt.xlabel(ax1_label, fontsize=18)
plt.ylabel(ax2_label, fontsize=18)
ax.set_title(plot_title, fontsize=18, fontweight='bold')
plt.savefig(f'./plots/{plot_title}.png')
plt.show()


