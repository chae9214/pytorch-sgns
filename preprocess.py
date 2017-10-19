import os
import re
import json
import pickle
from collections import Counter
from itertools import permutations
import matplotlib.pyplot as plt

def json_write(filename, obj):
    with open(filename, 'w') as fp:
        json.dump(obj, fp)

def json_read(filename):
    with open(filename, 'r') as fp:
        obj = json.load(fp)
    return obj

def pickle_write(filename, obj):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)

def pickle_read(filename):
    with open(filename, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def parse_kaggle_yummly(curr_len=0):
    filename = "data/dataset1.dat"

    if not os.path.exists(filename):
        print("Parsing Kaggle Yummly Data... ")
        food_dict = dict()
        data = json_read("data/kaggle_yummly.json")
        for i in range(len(data)):
            item = dict()
            item["title"] = ""
            item["cuisine"] = data[i]["cuisine"]
            item["ingredient"] = data[i]["ingredients"]
            food_dict[curr_len + i] = item
        pickle_write(filename, food_dict)
    else:
        food_dict = pickle_read(filename)
    return food_dict

def parse_recipe_dataset(curr_len=0):
    filename = "data/dataset2.dat"

    if not os.path.exists(filename):
        print("Parsing Recipe Dataset... ")
        food_dict = dict()
        data = open("data/recipe_dataset.ttl", 'r').read().split(".\n")
        for i in range(len(data)):
            str = data[i]
            match = re.search('(?<=#)\d+', str)
            if match:
                fid = curr_len + int(match.group(0)) - 1
            else:
                continue
            if fid not in food_dict:
                food_dict[fid] = dict()
                food_dict[fid]["title"] = ""
            if "food:cuisine" in str:
                food_dict[fid]["cuisine"] = re.findall('"(.*?)"', str)[0]
            elif "food:containsIngredient" in str:
                food_dict[fid]["ingredient"] = re.findall('"(.*?)"', str)
        pickle_write(filename, food_dict)
    else:
        food_dict = pickle_read(filename)
    return food_dict

def parse_yummly28k(curr_len=0):
    pass

def build_vocabulary(food_dict):
    indtoing_dict = dict()
    ingtoind_dict = dict()
    vocab_set = set()
    vocab_list = []
    vocab_len = []
    for fid, item in food_dict.items():
        vocab_set |= set(item["ingredient"])
        vocab_list += item["ingredient"]
        vocab_len.append(len(item["ingredient"]))
    vid = 0
    for v in vocab_set:
        indtoing_dict[vid] = v
        ingtoind_dict[v] = vid
        vid += 1
    vocab_count = Counter(vocab_list)
    vocab_len_count = Counter(vocab_len)
    print("average # of ingredients per dish: " + str(sum(vocab_len)/len(vocab_len)))
    return indtoing_dict, ingtoind_dict, vocab_set, vocab_count, vocab_len_count

SETSIZE_LIMIT = 20

def build_perm(food_dict, ingtoind):
    filename = "stat/permutation_dict.dat"

    if not os.path.exists(filename):
        print("Building Permutations... ")
        perm_dict = dict()
        for fid, item in food_dict.items():
            l = len(item["ingredient"]) - 1
            if l > 0 and l <= SETSIZE_LIMIT:
                inglist = item["ingredient"]
                indlist = [ingtoind[ing] for ing in inglist]
                if l not in perm_dict:
                    perm_dict[l] = []
                for i in range(l):
                    perm_dict[l].append((indlist[i], indlist[i+1:] + indlist[:i]))
                # for input, output in permutations(indlist, 2):
                #     perm_dict[l].append([input, output, indlist])
        pickle_write(filename, perm_dict)
    else:
        perm_dict = pickle_read(filename)
    return perm_dict

def item_frequency_graph(vocab_count):
    keys = list(map(lambda x: x[0], vocab_count.most_common(800)))
    values = list(map(lambda x: x[1], vocab_count.most_common(800)))
    plt.bar(range(len(keys)), values, align='center')
    #plt.xticks(range(len(keys)), list(keys))
    #plt.show()
    plt.savefig("stat/item_frequency_graph.png")

    ingword = []
    ingword_dict = dict()
    for ing in keys:
        ingword += ing.split()
        for word in ing.split():
            if word not in ingword_dict:
                ingword_dict[word] = []
            ingword_dict[word].append((ing, vocab_count[ing]))
    ingword_count = Counter(ingword)

    occ_len = []
    for word, occ in ingword_dict.items():
        occ_len.append(len(occ))
    print("average # of items a word occurs in: " + str(sum(occ_len) / len(occ_len)))
    return ingword_count, ingword_dict

def setsize_frequency_graph(vocab_len_count):
    sorted_values = []
    for l in sorted(vocab_len_count.keys()):
        sorted_values.append(vocab_len_count[l])
        print("# of sets with " + str(l) + " ingredients: " + str(vocab_len_count[l]))
    plt.bar(range(len(vocab_len_count)), sorted_values, align='center')
    plt.xticks(range(len(vocab_len_count)), sorted(vocab_len_count.keys()))
    #plt.show()
    plt.savefig("stat/setsize_frequency_graph.png")

if __name__ == '__main__':
    food_dict = dict()                                      # len = 96272
    food_dict1 = parse_kaggle_yummly()                      # len = 39774
    food_dict2 = parse_recipe_dataset(len(food_dict1))      # len = 56498
    food_dict.update(food_dict1)
    food_dict.update(food_dict2)

    indtoing_dict, ingtoind_dict, vocab_set, vocab_count, setsize_count = build_vocabulary(food_dict)
    pickle_write("stat/index_to_ingredient_dict.dat", indtoing_dict)
    pickle_write("stat/ingredient_to_index_dict.dat", ingtoind_dict)
    pickle_write("stat/vocab_count.dat", vocab_count.most_common())
    pickle_write("stat/vocab_set.dat", vocab_set)

    perm_dict = build_perm(food_dict, ingtoind_dict)

    ingword_count, ingword_dict = item_frequency_graph(vocab_count)
    setsize_frequency_graph(setsize_count)
    print("done")