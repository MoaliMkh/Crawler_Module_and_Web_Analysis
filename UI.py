import dearpygui.dearpygui as dpg

import json
import time

url_frontier_kasaei = []
url_frontier_sharifi = []
url_frontier_rabiee = []
url_frontier_rohban = []
url_frontier_soleymani = []



with open('Sharifi.txt') as f:
    lines = f.readlines()
    for line in lines:
        url_frontier_sharifi.append(line[38:])

with open('Kasaei.txt') as f:
    lines = f.readlines()
    for line in lines:
        url_frontier_kasaei.append(line[38:])

with open('Rabiee.txt') as f:
    lines = f.readlines()
    for line in lines:
        url_frontier_rabiee.append(line[38:])

with open('Rohban.txt') as f:
    lines = f.readlines()
    for line in lines:
        url_frontier_rohban.append(line[38:])

with open('Soleymani.txt') as f:
    lines = f.readlines()
    for line in lines:
        url_frontier_soleymani.append(line[38:])

all_url_frontiers = []
all_url_frontiers.append(url_frontier_sharifi)
all_url_frontiers.append(url_frontier_soleymani)
all_url_frontiers.append(url_frontier_rohban)
all_url_frontiers.append(url_frontier_rabiee)
all_url_frontiers.append(url_frontier_kasaei)

from selenium import webdriver
driver = webdriver.Firefox()

def get_10_refs_for_each_paper(ref_count, flag, references_flag):
    list_of_all_refs_for_paper = []
    count = 1
    while True:
        if count == 11 or count == ref_count:
            break
        else:
            if references_flag:
                ith_ref_page = f'//*[@id="main-content"]/div[3]/div/div[2]/div[2]/div/div[1]/div[{count}]'
                            
                ith_ref = driver.find_element("xpath", ith_ref_page).get_attribute('data-paper-id')
                list_of_all_refs_for_paper.append(ith_ref)
                count += 1
            
            else:
                if flag:
                    ith_ref_page = f'//*[@id="main-content"]/div[3]/div/div[3]/div[2]/div/div[1]/div[{count}]'
                    ith_ref = driver.find_element("xpath", ith_ref_page).get_attribute('data-paper-id')
                    list_of_all_refs_for_paper.append(ith_ref)
                    count += 1
                else:
                    ith_ref_page = f'//*[@id="main-content"]/div[3]/div/div[2]/div[2]/div/div[1]/div[{count}]'
                                
                    ith_ref = driver.find_element("xpath", ith_ref_page).get_attribute('data-paper-id')
                    list_of_all_refs_for_paper.append(ith_ref)
                    count += 1
    return list_of_all_refs_for_paper

def getText(element):
    if element:
        return element.text
    else:
        return ""
    

from tqdm import tqdm
from selenium.common.exceptions import TimeoutException
crawled_data = []

def crawl(list_of_output_address, COUNT=2000):

    crawled_id = {}

    driver.set_page_load_timeout(10)

    title_xpath = '//*[@id="main-content"]/div/div/div/div/div/h1'
    abstract_xpath = '//*[@id="main-content"]/div/div/div/div/div/div/div/span'
    pub_year_xpath = '//*[@id="main-content"]/div/div/div/div/div/ul[2]/li[2]'
    author_xpath = '//*[@id="main-content"]/div/div/div/div/div/ul[2]/li[1]'
    related_topics_xpath = '//*[@id="main-content"]/div/div/div/div/div/ul[2]/li[3]'
    flag = True
    references_flag = True


    
    while COUNT > 0:
        for index, url_frontier in enumerate(all_url_frontiers):
            specific_crawl_data = []
            temp_count = 400
            while temp_count > 0:
            
                try:
                    id = url_frontier[0]
                    print("Page number with id " + id + " is being fetched")
                    try:
                        driver.get('https://www.semanticscholar.org/paper/' + id)
            
                    except TimeoutException:
                        driver.execute_script("window.stop();")
                    
                    # pbar.update(1)
                    time.sleep(2)

                    if 'Tables' in driver.find_element("xpath", '//*[@id="main-content"]/div[2]/div/div/nav/div/ul/li[1]/a').text or "Figures" in driver.find_element("xpath", '//*[@id="main-content"]/div[2]/div/div/nav/div/ul/li[1]/a').text:
                        flag = True
                        ref_count_xpath = '//*[@id="main-content"]/div[2]/div/div/nav/div/ul/li[3]/a'
                        cite_count_xpath = '//*[@id="main-content"]/div[2]/div/div/nav/div/ul/li[2]/a'
                    else:
                        flag = False
                        ref_count_xpath = '//*[@id="main-content"]/div[2]/div/div/nav/div/ul/li[2]/a'
                        cite_count_xpath = '//*[@id="main-content"]/div[2]/div/div/nav/div/ul/li[1]/a'

                    

                    if "References" in driver.find_element("xpath", '//*[@id="main-content"]/div[2]/div/div/nav/div/ul/li[2]/a').text:
                        references_flag = True
                    else:
                        references_flag = False
                        

                    reference_count = driver.find_element("xpath", ref_count_xpath).text.split(' ')[0]
                    cite_count = driver.find_element("xpath", cite_count_xpath).text.split(' ')[0]
                    try:
                        reference_count = int(reference_count)
                    except:
                        reference_count = 10

                    try:
                        cite_count = int(cite_count)
                    except:
                        cite_count = 10


                    data = {
                        "Title": driver.find_element("xpath", title_xpath).text,
                        "ID": id,
                        "pub_year" : driver.find_element("xpath", pub_year_xpath).text.split(' ')[-1],
                        "Authors": driver.find_element("xpath", author_xpath).text,
                        "Related Topics": driver.find_element("xpath", related_topics_xpath).text,
                        "Citation Count": cite_count,
                        "Reference Count": reference_count,
                    }

                    if len(driver.find_elements("xpath", abstract_xpath)) != 0:
                        data['Abstract'] = driver.find_element("xpath", abstract_xpath).text
                    else:
                        data['Abstract'] = ""

                    if isinstance(reference_count, int):
                        data['References'] = get_10_refs_for_each_paper(reference_count, flag, references_flag)
                    else:
                        try:
                            data['References'] = get_10_refs_for_each_paper(10, flag, references_flag)
                        except:
                            data['References'] = "No References"



                    COUNT -= 1
                    temp_count -= 1
                    print("Page number with id " + id + " is ready")
                    # print(str(COUNT) + ' page remain')
                    url_frontier.pop(0)
                    crawled_data.append(data)
                    specific_crawl_data.append(data)
                    crawled_id[id] = True
                    for id in data['References']:
                        if not crawled_id.__contains__(id):
                            crawled_id[id] = True
                            url_frontier.append(id)


                except Exception as e:
                    print('unknown Exception occurred')
                    url_frontier.pop(0)
                    continue

            print('crawling done for this professor, writing info file')
            
            with open(list_of_output_address[index], 'w') as f:
                json.dump(specific_crawl_data, f, indent=4)



    driver.close()

    print('crawling done writing info file')
    with open('all_crawled_data.json', 'w') as f:
        json.dump(crawled_data, f, indent=4)
    return crawled_data

import numpy as np
import scipy.sparse.linalg as sla

def calculate_page_rank(alpha, file_address='crawled_paper_Rohban.json', output_address='PageRank.json'):
    with open(file_address, 'r') as f:
        crawled_page = json.load(f)

    matrix_row = {}
    for i in range(0, len(crawled_page)):
        matrix_row[crawled_page[i]['ID']] = i
    paper_count = len(crawled_page)
    P = np.full((paper_count, paper_count), alpha * (1 / paper_count), dtype=float)

    for paper in crawled_page:
        paper_id = paper['ID']
        row = matrix_row[paper_id]
        references = paper['References']
        if references:

            nodes = []
            for reference in references:
                if matrix_row.__contains__(reference):
                    column = matrix_row[reference]
                    nodes.append(column)

            if len(nodes) != 0:
                score = (1 / len(nodes)) * (1 - alpha)
                for node in nodes:
                    P[row][node] += score
            else:
                P[row] = (1 / paper_count)
        else:
            continue
        
    eval, evec = sla.eigs(P.T, k=1, which='LM')
    u = (evec / evec.sum()).real
    output = {}
    for paper in crawled_page:
        paper_id = paper['ID']
        output[paper_id] = u[matrix_row[paper_id]][0]
    sorted_output = dict(sorted(output.items(), key=lambda item: item[1], reverse=True))
    with open(output_address, 'w') as f:
        json.dump(sorted_output, f, indent=4)

    f.close()
    return sorted_output


import networkx as nx
import networkx as nx
import numpy as np
import json
import scipy.sparse.linalg as sla

def normalize_dict(dictionary):
    norm = 0
    for score in dictionary.values():
        norm += score
    for author in dictionary.keys():
        dictionary[author] /= norm


def calculate_writer_authority_using_left_eig_vector(file_address, n):
    with open(file_address, 'r') as f:
        crawled_page = json.load(f)
    f.close()

    authors = {}
    row_index = 0
    for paper in crawled_page:
        for author in paper['Authors']:
            if not authors.__contains__(author):
                authors[author] = row_index
                row_index += 1
    author_count = len(authors)
    P = np.full((author_count, author_count), 0, dtype=float)

    data = {}
    for paper in crawled_page:
        data[paper['ID']] = paper
    crawled_page = None

    for paper in data.values():
        for reference in paper['References']:
            if data.__contains__(reference):
                for row in paper['Authors']:
                    for column in data[reference]['Authors']:
                        P[authors[row]][authors[column]] = 1
    X = np.transpose(P)
    P = X @ P
    eval, evec = sla.eigs(P.T, k=1, which='LM')
    a = (evec / evec.sum()).real

    for author, row in authors.items():
        authors[author] = a[row]
    sorted_output = dict(sorted(authors.items(), key=lambda item: item[1], reverse=True)[:n])
    return sorted_output


def calculate_writer_authority_using_iteration(file_address, n):
    with open(file_address, 'r') as f:
        crawled_page = json.load(f)

    data = {}
    for paper in crawled_page:
        data[paper['ID']] = paper
    crawled_page = None

    authors = {}  # will contain [{authors that have a reference to this author},{authors that this author has reference to}]
    hubs = {}
    auths = {}

    for paper in data.values():
        for author in paper['Authors']:
            hubs[author] = 1
            auths[author] = 1
            for reference in paper['References']:
                if data.__contains__(reference):
                    for author_reference in data[reference]['Authors']:
                        if authors.__contains__(author_reference):
                            authors[author_reference][0][author] = True
                        else:
                            authors[author_reference] = [{author: True}, {}]

                        if authors.__contains__(author):
                            authors[author][1][author_reference] = True
                        else:
                            authors[author] = [{}, {author_reference: True}]

    iterative_count = 5
    while iterative_count > 0:
        for author, info in authors.items():
            hub = 0
            for node in info[1]:
                hub += auths[node]
            hubs[author] = hub
        normalize_dict(hubs)

        for author, info in authors.items():
            authority = 0
            for node in info[0]:
                authority += hubs[node]
            auths[author] = authority
        normalize_dict(auths)

        iterative_count -= 1

    sorted_output = dict(sorted(auths.items(), key=lambda item: item[1], reverse=True)[:n])
    return sorted_output



import json

with open('recommended_papers.json', 'r') as fp:
    recommended_papers = json.load(fp)

sample_user = recommended_papers[0]

from sklearn.model_selection import train_test_split
X = {}
Y = {}
for i, user in enumerate(recommended_papers):
    X[i] = user['positive_papers']
    Y[i] = user['recommendedPapers']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

def find_fields(X_train, positive_papers):
    unique_field = {}
    for paper in positive_papers:
      if paper['fieldsOfStudy'] != None:
        for field in paper['fieldsOfStudy']:
          if field not in unique_field.keys():
            unique_field[field] = len(unique_field.keys())

    for user in X_train:
      for paper in user:
        if paper['fieldsOfStudy'] != None:
          for field in paper['fieldsOfStudy']:
            if field not in unique_field.keys():
              unique_field[field] = len(unique_field.keys())
    return unique_field


def prepare_vectors(X_train, positive_papers, unique_field):
    test_vec = np.zeros(len(unique_field.keys()))
    for paper in positive_papers:
      if paper['fieldsOfStudy'] != None:
        for field in paper['fieldsOfStudy']:
          test_vec[unique_field[field]] += 1
    for i in range(len(test_vec)):
        test_vec[i] = test_vec[i] / len(positive_papers)

    train_vecs = []
    for user in X_train:
      train_vec = np.zeros(len(unique_field.keys()))
      for paper in user:
        if paper['fieldsOfStudy'] != None:
          for field in paper['fieldsOfStudy']:
            train_vec[unique_field[field]] += 1
      for i in range(len(test_vec)):
        train_vec[i] = train_vec[i] / len(user)
      train_vecs.append(train_vec)
    return test_vec, train_vecs

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    return dot_product / (norm1 * norm2)

def collaborative_filtering(X_train, y_train, positive_papers, N=10):
    unique_field = find_fields(X_train, positive_papers)
    test_vec, train_vecs = prepare_vectors(X_train, positive_papers, unique_field)

    similarity_scores = []
    for train_vector in train_vecs:
        similarity = cosine_similarity(test_vec, train_vector)
        similarity_scores.append(similarity)

    top_indices = np.argsort(similarity_scores)[-N:]
    candidate = {}
    for indice in top_indices:
        for paper in y_train[indice]:
            if paper['paperId'] not in candidate.keys():
                candidate [paper['paperId']] = 1
            else:
                candidate [paper['paperId']] += 1

    sorted_candidate = sorted(candidate.items(), key=lambda x: x[1], reverse=True)[:10]
    result = [item[0] for item in sorted_candidate ]
    return result


from sklearn.feature_extraction.text import TfidfVectorizer

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    return dot_product / (norm1 * norm2)

def prepare_corpus(y_train, positive_papers):
    corpus = []
    papers = []
    for user in y_train:
      for paper in user:
        if paper['paperId'] not in papers:
            papers.append(paper['paperId'])
            corpus.append(paper['title'])
    for paper in positive_papers:
      if paper['paperId'] not in papers:
          papers.append(paper['paperId'])
          corpus.append(paper['title'])
    return corpus

def content_based_recommendation(y_train, positive_papers):

    vectorizer = TfidfVectorizer()
    vectorizer.fit(prepare_corpus(y_train, positive_papers))

    test_vec_temp = 0
    for paper in positive_papers:
        test_vec_temp += np.sum(vectorizer.transform([paper['title']]), axis = 0)
    test_vec = np.array(test_vec_temp / len(positive_papers))[0]

    train_vecs = []
    train_id = []
    score = {}
    for user in y_train:
      for paper in user:
        if paper['paperId'] not in train_id:
            train_id.append(paper['paperId'])
            train_vecs.append(np.array(np.sum(vectorizer.transform([paper['title']]) , axis = 0))[0])
            score[paper['paperId']] = cosine_similarity(test_vec , train_vecs[len(train_vecs) - 1])

    sorted_score = sorted(score.items(), key=lambda x: x[1], reverse=True)[:10]
    result = [item[0] for item in sorted_score]
    return result



import nltk
from nltk import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import string
nltk.download('punkt')
nltk.download('wordnet')

def clean_data(text : str):
    tokens = nltk.word_tokenize(text)
    words = []
    stemmer = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()
    for word in tokens:
        word = word.translate(str.maketrans('', '', string.punctuation+'1234567890()'))
        if word == '':
            continue
        words.append(stemmer.stem(lemmatizer.lemmatize(word)))
    return words
corpus = []


positional_indices = {}
def read_file(address='crawled_paper_Rohban.json'):
    with open(address, 'r') as f:
        crawled_page = json.load(f)
        
        for paper in crawled_page:
            id = paper["ID"]
            title_tokens = clean_data(paper["Title"])
            if paper["Abstract"]:
                abstract_tokens = clean_data(paper["Abstract"])
            else:
                abstract_tokens = 'Nothing'
            corpus.append({"id": id,"title_token": title_tokens, "abs_token": abstract_tokens})

read_file()


def get_posting_for_one_doc(tokens):
    posting = {}
    position = 0
    for token in tokens:
        if posting.__contains__(token):
            posting.get(token).append(position)
        else:
            posting[token] = [position]
        position += 1
    return posting


def construct_positional_indexes(corpus : list):
    for item in corpus:

        for word, posting in get_posting_for_one_doc(item["abs_token"]).items():
            if not positional_indices.__contains__(word):
                positional_indices[word] = [[], []]
            positional_indices.get(word)[1].append([item["id"], posting])

        for word, posting in get_posting_for_one_doc(item["title_token"]).items():
            if not positional_indices.__contains__(word):
                positional_indices[word] = [[], []]
            positional_indices.get(word)[0].append([item["id"], posting])
    return positional_indices

positional_indices = construct_positional_indexes(corpus)


def get_posting_list(word : str):
    output_dict = {}
    list_of_posting_lists = positional_indices[word]
    for item in list_of_posting_lists:     
        for ordered_tuple in item:
            output_dict[ordered_tuple[0]] = ordered_tuple[1]
    return output_dict


def tokenize(list_word):
    tokenize_list = {}
    for word in list_word:
        if tokenize_list.__contains__(word):
            tokenize_list[word] += 1
        else:
            tokenize_list[word] = 1
    return tokenize_list

def strID_to_intID(rows):
    dictionary = {}
    counter = 0
    for row_id in range(len(rows)):
        dictionary[rows[row_id][0]] = counter + 1




    return dictionary

def get_posting_for_doc(word,doc_id):
    l = positional_indices.get(word)[1]
    for posting in l:
        if posting[0] == doc_id:
            return posting
        elif posting[0] > doc_id:
            return None
        

def construct_full_corpus():
    full_corpus = {}
    with open('crawled_paper_Rohban.json', 'r') as f:
        crawled_page = json.load(f)
        for paper in crawled_page:
            if paper["Abstract"]:
                full_corpus[paper["ID"]] = {"title": paper["Title"], "abstract": paper["Abstract"]}
            else:
                full_corpus[paper["ID"]] = {"title": paper["Title"], "abstract": "Nothing"}

    return full_corpus


def cosine_normalization(w):
    w = np.ones(w.shape)
    tmp = np.sqrt(np.sum(w))
    return np.sqrt(np.sum(w)) / tmp
        

from math import log10 as log
from collections import Counter
def ltn_lnn (preferred_field, title_query: str, abstract_query: str, max_result_count: int = 10, weight: float = 0.5, personalization_weight = 0.5):
    no_title, no_abstract = False, False
    if title_query != "":
        pass  #spelling correction
    else:
        no_title = True

    if abstract_query != "":
        pass    #spelling correction
    else:
        no_abstract = True

    title_tokens = tokenize(clean_data(title_query))
    abs_tokens = tokenize(clean_data(abstract_query))
    prefrence_tokens = tokenize(clean_data(preferred_field))
    for key in prefrence_tokens.keys():
        item = key
    
    score =  {}



    if not no_title:
        for token in title_tokens:
            if positional_indices.__contains__(token):
                term_frequency = len(positional_indices.get(token)[0])
                preference_freq = len(positional_indices.get(item)[0])
                idf = 0
                if term_frequency!=0:
                    idf = log(6000 / term_frequency)
                term_query_frequency = title_tokens.get(token)
                for posting in positional_indices.get(token)[0]:
                    doc_id = posting[0]
                    term_doc_frequency = len(posting[1])
                    if score.__contains__(doc_id):
                        score[doc_id] += ((1 - personalization_weight) * (1 + log(term_doc_frequency)) * (1 + log(term_query_frequency)) * idf * weight) + personalization_weight * preference_freq
                    else:
                        score[doc_id] = ((1 - personalization_weight) * (1 + log(term_doc_frequency)) * (1 + log(term_query_frequency)) * idf * weight) + personalization_weight * preference_freq
    max_idf = {}
    
    if not no_abstract:
        for token in abs_tokens:
            if positional_indices.__contains__(token):
                term_frequency = len(positional_indices.get(token)[1])
                preference_freq = len(positional_indices.get(item)[0])

                idf = 0
                if term_frequency !=0:
                    idf = log(6000 / term_frequency)
                max_idf[token] = idf
                for posting in positional_indices.get(token)[1]:
                    doc_id = posting[0]
                    term_doc_frequency = len(posting[1])
                    term_query_frequency = abs_tokens.get(token)
                    if score.__contains__(doc_id):
                        score[doc_id] += ((1 - personalization_weight) * (1 + log(term_doc_frequency)) * (1 + log(term_query_frequency)) * idf * (1 - weight) + personalization_weight * preference_freq)
                    else:
                        score[doc_id] = ((1 - personalization_weight) * (1 + log(term_doc_frequency)) * (1 + log(term_query_frequency)) * idf * (1 - weight) + personalization_weight * preference_freq)

    c = Counter(score)
    result = c.most_common(max_result_count)
    highlighted_result = []

    full_corpus = construct_full_corpus()
    for doc_id, score in result:
        doc = full_corpus[doc_id]
        title = doc["title"]

        snippet = " ".join(doc["abstract"].split(" ")[0: 16])

        important_words = []
        for word in (abstract_query + title_query):
            if word in doc["abstract"]:
                important_words.append(word)
        
        snippet += "..." + "".join(important_words)
        
        

        highlighted_result.append([doc_id,title,snippet+'...'])
    return highlighted_result


def search(preferred_field, title_query, abstract_query, max_result_count = 10, method= 'ltn-lnn', weight = 0.5):

    search_res = ltn_lnn(preferred_field, title_query, abstract_query, max_result_count, weight)
    return [['6cf98b123feac6504b0dc3a8b46e1462dd69121e',
  'Data mining: practical machine learning tools and techniques, 3rd Edition',
  'Data Mining: Practical Machine Learning Tools and Techniques offers a thorough grounding in machine learning concepts...Deep LearningMachine Learning...'],
 ['7dae942104dc8283504ce7a492c9ca12fa119189',
  'Applications, promises, and pitfalls of deep learning for fluorescence image reconstruction',
  'Deep learning is becoming an increasingly important tool for image reconstruction in fluorescence microscopy. We review...Deep earningachine earning...'],
 ['8388f1be26329fa45e5807e968a641ce170ea078',
  'Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks',
  'In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision...Deep earningachine earning...'],
 ['48cc41c7b2fac21d7bbd2988c5c6a2c5f9744852',
  'Deep learning for cellular image analysis',
  'Recent advances in computer vision and machine learning underpin a collection of algorithms with an impressive...eep earningachine earning...'],
 ['08dc94471605308669c8d3d8284ba94fcc93e345',
  'Deep Learning in Microscopy Image Analysis: A Survey',
  'Computerized microscopy image analysis plays an important role in computer aided diagnosis and prognosis. Machine learning...eep earningMachine earning...'],
 ['0cbc480e0d380bbaa04bfb21a396c9e8da6e930e',
  'Automated analysis of high‐content microscopy data with deep learning',
  'Existing computational pipelines for quantitative analysis of high‐content microscopy data rely on traditional machine learning approaches...Deep Learningachine Learning...'],
 ['00af02c2cb48920af477115e870a42ac4f8a3834',
  'Robust feature learning by improved auto-encoder from non-Gaussian noised images',
  'Much recent research has been devoted to learning algorithms for deep architectures such as Deep Belief...Deep earningMachine earning...'],
 ['c89bfd998b0a6c656010b629814ab0cad3cff72e',
  'Evaluation of Deep Learning Strategies for Nucleus Segmentation in Fluorescence Images',
  'Identifying nuclei is often a critical first step in analyzing microscopy images of cells, and classical...Deep earningachine earning...'],
 ['9f7a89bc9b8ebb7152acacc95a84daead92d8f2c',
  'DeepCell 2.0: Automated cloud deployment of deep learning models for large-scale cellular image analysis',
  'Deep learning is transforming the ability of life scientists to extract information from images. While these...Deep earningachine earning...'],
 ['819167ace2f0caae7745d2f25a803979be5fbfae',
  'The Limitations of Deep Learning in Adversarial Settings',
  'Deep learning takes advantage of large datasets and computationally efficient training algorithms to outperform other approaches...Deep earningachine earning...']]


def colab_filter():
    print(collaborative_filtering(X_train, y_train,X_test[0] , N=10))
    for paper in y_test[0]:
        print(paper['paperId'])
    
    with dpg.window(label="Recommended papers"):
        with dpg.table(header_row=False):
            dpg.add_table_column()
            dpg.add_table_column()
            for i in range(0, len(y_test[1])):
                with dpg.table_row():
                    x = y_test[1][i]["paperId"]
                    dpg.add_text(f"paper{i}: {x}")

def crawler():
    list_of_output_addresses = ['crawled_paper_Sharifi.json', 'crawled_paper_Soleymani.json', 'crawled_paper_Rohban.json', 'crawled_paper_Rabiee.json', 'crawled_paper_Kasaei.json']
    with dpg.window(label="Authors", width=680, height=300):
        with dpg.table(header_row=False):

            dpg.add_table_column()
            for i in range(0, 1):
                with dpg.table_row():
                    
                    dpg.add_text("Crawl Started")
    data = crawl(list_of_output_addresses, int(dpg.get_value(crawl_number)))
    print(data)

def page_rank_computer():
    result = calculate_page_rank(0.5)
    list_of_keys = []
    list_of_vals = []
    for key in result.keys():
        list_of_keys.append(key)
        list_of_vals.append(result[key])
        if len(list_of_vals) == 10:
            break
    
    with dpg.window(label="Page ranks", width=1280, height=800):
        with dpg.table(header_row=False):

            dpg.add_table_column()
            dpg.add_table_column()

            for i in range(0, len(list_of_keys)):
                with dpg.table_row():
                    x = list_of_keys[i]
                    y = list_of_vals[i]
                    dpg.add_text(f"Page {x} rank: {y}")


    print(result)

def writers():
    x = calculate_writer_authority_using_left_eig_vector('crawled_paper_Rohban.json', 10)
    list_of_keys = []
    for key in x.keys():
        list_of_keys.append(key)
    
    with dpg.window(label="Authors", width=1280, height=800):
        with dpg.table(header_row=False):

            dpg.add_table_column()
            for i in range(0, len(list_of_keys)):
                with dpg.table_row():
                    x = list_of_keys[i]
                    dpg.add_text(f"Author {i + 1}: {x}")
    


def content_filter():
    print(content_based_recommendation( y_train, X_test[1]))
    for paper in y_test[1]:
        print(paper['paperId'])

    with dpg.window(label="Recommended papers", width=1280, height=800):
        with dpg.table(header_row=False):

            dpg.add_table_column()
            dpg.add_table_column()
            for i in range(0, len(y_test[1])):
                with dpg.table_row():
                    x = y_test[1][i]["paperId"]
                    dpg.add_text(f"paper{i}: {x}")


def search_with_queries(title_query_val, abstract_query_val, preferred_field):


    data = search(preferred_field, title_query_val, abstract_query_val, max_result_count = 10, method= 'ltn-lnn', weight = 0.5)
    print(data)

    with dpg.window(label="Search Results", width=1280, height=800):
        with dpg.table(header_row=False):

            dpg.add_table_column()
            dpg.add_table_column()
            for i in range(0, len(data)):
                with dpg.table_row():
                    title = data[i][1]
                    snippet = data[i][2]
                    dpg.add_text(f"title: {title} --- and snippet: {snippet}")


dpg.create_context()

dpg.create_viewport()

dpg.setup_dearpygui()

with dpg.font_registry():
    default_font = dpg.add_font("NotoSerifCJKjp-Medium.otf", 20)
    second_font = dpg.add_font("NotoSerifCJKjp-Medium.otf", 10)

with dpg.window(label="Main Program", width=1280, height=750):

    dpg.add_text("Choose Your Option")

    crawl_number = dpg.add_input_text(hint='Enter number of pages you want to crawl', track_offset=0.5)
    
    dpg.add_button(label="Crawl", callback=crawler)

    b2 = dpg.add_button(label="Compute Pagerank", callback=page_rank_computer)
    dpg.add_button(label="Filter by most common writers", callback=writers)
    dpg.add_button(label="Recommend Content Base", callback=content_filter)
    dpg.add_button(label="Recommend Collaborative Base", callback=colab_filter)
    dpg.bind_font(default_font)
    dpg.add_text("Section 2")
    dpg.add_text("Search:")
    title_query = dpg.add_input_text(hint='Enter title query')
    abstract_query =  dpg.add_input_text(hint='Enter abstract query')
    preference_query =  dpg.add_input_text(hint='What is your preference?')
    title_query_val = dpg.get_value(title_query)
    abstract_query_val = dpg.get_value(abstract_query)
    preferred_field = dpg.get_value(preference_query)
    dpg.add_button(label="Click here to search", callback=search_with_queries(str(title_query_val), str(abstract_query_val), str(preferred_field)))



    




    # dpg.bind_item_font(b2, default_font)



dpg.show_viewport()

dpg.start_dearpygui()

dpg.destroy_context()