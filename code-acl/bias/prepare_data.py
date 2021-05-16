import pickle
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit

pomt_accepted_labels = ['pants on fire!', 'false', 'mostly false', 'half-true', 'mostly true', 'true']
snes_accepted_labels = ['false', 'mostly false', 'mixture', 'mostly true', 'true']

def read_snippets(file_path):
    titles = ["filler" for _ in range(10)]
    snippets = ["filler" for _ in range(10)]

    if not os.path.exists(file_path):
        return titles, snippets, False

    with open(file_path, "r", encoding="utf8") as f:
        lines = f.readlines()
        for line_i, line in enumerate(lines):
            content = line.split("\t")
            title = content[1]
            snippet = content[2]
            rank = int(content[0])

            if line_i > 9:
                continue # only happens once due to dataset mistake.
            if len(title) > 2:
                titles[line_i] = title
            if len(snippet) > 2:
                snippets[line_i] = snippet

    return titles, snippets, True

def prepare(dataset_path, file):
    # save snippets like [claimID, s1, s2, ..., s10]

    df = pd.read_csv(dataset_path + file + ".tsv", sep="\t", header=None)
    vals = df.values

    all_rows = []
    missing_claimID_snippets = []

    for i in range(len(vals)):
        claimID = vals[i, 0]
        _, snippets, exists = read_snippets(dataset_path + "snippets/" + claimID)
        row = [claimID] + snippets

        all_rows.append(row)

        if not exists:
            missing_claimID_snippets.append(claimID)

    df = pd.DataFrame(all_rows)
    df.to_csv(dataset_path + file + "_snippets.tsv", sep="\t", header=None, index=None)
    pickle.dump(missing_claimID_snippets, open(dataset_path + file + "_missingSnippets.pkl", "wb"))
    print(df.shape)

def merge(path, type1, type2):

    df1 = pd.read_csv(path + type1 + ".tsv", sep="\t", header=None)
    df2 = pd.read_csv(path + type2 + ".tsv", sep="\t", header=None)

    df = pd.concat((df1, df2))
    print(df1.shape, df2.shape, df.shape)

    df1 = pd.read_csv(path + type1 + "_snippets.tsv", sep="\t", header=None)
    df2 = pd.read_csv(path + type2 + "_snippets.tsv", sep="\t", header=None)

    df_snippets = pd.concat((df1, df2))
    print(df1.shape, df2.shape, df_snippets.shape)

    p1 = pickle.load(open(path + type1 + "_missingSnippets.pkl", "rb"))
    p2 = pickle.load(open(path + type2 + "_missingSnippets.pkl", "rb"))

    missing_claimID_snippets = p1 + p2
    print(len(p1), len(p2), len(missing_claimID_snippets))

    df.to_csv(path + "all" + ".tsv", sep="\t", header=None, index=None)
    df_snippets.to_csv(path + "all" + "_snippets.tsv", sep="\t", header=None, index=None)
    pickle.dump(missing_claimID_snippets, open(path + "all" + "_missingSnippets.pkl", "wb"))

def make_subset(path, origin, accepted_labels):

    all = pd.read_csv(path + "all" + ".tsv", sep="\t", header=None).values
    snippets = pd.read_csv(path + "all" + "_snippets.tsv", sep="\t", header=None).values
    missing_claimID_snippets = set(pickle.load(open(path + "all" + "_missingSnippets.pkl", "rb")))

    chosen_all = []
    chosen_snippets = []

    for i in range(len(all)):
        claimID = all[i][0]
        claimLabel = all[i][2]
        if claimID.split("-")[0] == origin and claimID not in missing_claimID_snippets and claimLabel in accepted_labels:
            chosen_all.append(all[i])
            chosen_snippets.append(snippets[i])

    chosen_all_df = pd.DataFrame(chosen_all)
    chosen_snippets_df = pd.DataFrame(chosen_snippets)
    chosen_all_df.to_csv(path + origin + "/" + origin + ".tsv", sep="\t", header=None, index=None)
    chosen_snippets_df.to_csv(path + origin + "/" + origin + "_snippets.tsv", sep="\t", header=None, index=None)
    pickle.dump(accepted_labels, open(path + origin + "/" + origin + "_labels.pkl", "wb"))

    print(chosen_all_df.shape, chosen_snippets_df.shape)

    all_labels = chosen_all_df.values[:, 2]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_index, test_index in sss.split(all_labels, all_labels):
        pass

    train_labels = all_labels[train_index]
    org_train_index = train_index
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1.0/8.0, random_state=0)
    for train_index, val_index in sss.split(train_labels, train_labels):
        pass

    val_index = org_train_index[val_index]
    train_index = org_train_index[train_index]

    pickle.dump([train_index, val_index, test_index], open(path + origin + "/" + origin + "_index_split.pkl", "wb"))

if __name__ == '__main__':
    multi_fc_path = "../../multi_fc_publicdata/"

    prepare(multi_fc_path, "train")
    prepare(multi_fc_path, "dev")

    # merge train and dev to have control over how they were created
    merge(multi_fc_path, "train", "dev")

    # extract specific sub-datasets
    make_subset(multi_fc_path, "pomt", pomt_accepted_labels)
    make_subset(multi_fc_path, "snes", snes_accepted_labels)