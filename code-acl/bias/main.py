import sys
import os
sys.path.append('../../code-acl')
sys.path.append(os.getcwd())
os.environ['OMP_NUM_THREADS'] = "1"
import argparse
import pandas as pd
import pickle
from model.generator import TransformerDataset, transformer_collate
from model.bertmodel import MyBertModel
from model.lstmmodel import LSTMModel
import torch
from parameters import BERT_MODEL_PATH, CLAIM_ONLY, CLAIM_AND_EVIDENCE, EVIDENCE_ONLY, DEVICE, INPUT_TYPE_ORDER
from transformers import AdamW
import numpy as np
from utils.utils import print_message, clean_str
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from torchnlp.word_to_vector import GloVe
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from hypopt import GridSearch
from tqdm import tqdm

def load_data(dataset):
    path = "../../multi_fc_publicdata/" + dataset + "/"
    main_data = pd.read_csv(path + dataset + ".tsv", sep="\t", header=None)
    snippets_data = pd.read_csv(path + dataset + "_snippets.tsv", sep="\t", header=None)
    label_order = pickle.load(open(path + dataset + "_labels.pkl", "rb"))
    splits = pickle.load(open(path + dataset + "_index_split.pkl", "rb"))

    return main_data, snippets_data, label_order, splits

def make_generators(main_data, snippets_data, label_order, splits, params, dataset_generator=TransformerDataset, other_dataset=False):
    generators = []

    all_labels = main_data.values[:,2]
    counter = Counter(all_labels)
    ss = ""
    for c in label_order:
        ss = ss + ", " + str(c) + " (" + str(np.around(counter[c]/len(all_labels) * 100,1)) + "\%)"
        #print(c, np.around(counter[c]/len(all_labels) * 100,1), "%", counter[c])
    print("len", len(all_labels), ss)

    for isplit, split in enumerate(splits):
        sub_main_data = main_data.values[split]
        sub_snippets_data = snippets_data.values[split]

        tmp = dataset_generator(sub_main_data, sub_snippets_data, label_order)
        if isplit == 0:
            generator = torch.utils.data.DataLoader(tmp, **params[0])
        else:
            generator = torch.utils.data.DataLoader(tmp, **params[1])

        generators.append(generator)

    # make class weights
    labels = main_data.values[splits[0]][:,2]
    labels = np.array([label_order.index(v) for v in labels])


    if not other_dataset:
        label_weights = torch.tensor(compute_class_weight("balanced", classes=np.arange(len(label_order)), y=labels).astype(np.float32))
    else:
        label_weights = None

    return generators[0], generators[1], generators[2], label_weights

def evaluate(generator, model, other_from=None, ignore_snippet=None):
    all_labels = []
    all_predictions = []

    all_claimIDs = []
    all_logits = []

    for vals in generator:
        claimIDs, claims, labels, snippets = vals[0], vals[1], vals[2], vals[3]

        if ignore_snippet is not None:
            for i in range(len(snippets)):
                snippets[i][ignore_snippet] = "filler"

        all_labels += labels
        logits = model(claims, snippets)

        predictions = torch.argmax(logits, 1).cpu().numpy()

        if other_from == "pomt": # other data is pomt, and model is trained on snes
            # this case is fine
            pass
        elif other_from == "snes": # other data is snes, and model is trained on pomt
            # in this case both "pants on fire!" and "false" should be considered as false
            predictions[predictions == 0] = 1 # 0 is "pants on fire!" and 1 is "false" for pomt.

        all_predictions += predictions.tolist()

        all_claimIDs += claimIDs
        all_logits += logits.cpu().numpy().tolist()

    f1_micro = f1_score(all_labels, all_predictions, average="micro")
    f1_macro = f1_score(all_labels, all_predictions, average="macro")

    return f1_micro, f1_macro, all_claimIDs, all_logits, all_labels, all_predictions

def train_step(optimizer, vals, model, criterion):
    optimizer.zero_grad()

    claimIDs, claims, labels, snippets = vals[0], vals[1], torch.tensor(vals[2]).to(DEVICE), vals[3]

    logits = model(claims, snippets)
    loss = criterion(logits, labels)

    loss.backward()
    optimizer.step()

    return loss


def get_embedding_matrix(generators, dataset, min_occurrence=1):
    savename = "preprocessed/" + dataset + "_glove.pkl"
    if os.path.exists(savename):
        tmp = pickle.load(open(savename, "rb"))
        glove_embedding_matrix = tmp[0]
        word2idx = tmp[1]
        idx2word = tmp[2]
        return glove_embedding_matrix, word2idx, idx2word

    glove_vectors = GloVe('840B')
    all_claims = []
    all_snippets = []
    for gen in generators:
        for vals in gen:
            claims = vals[1]
            claims = [clean_str(v) for v in claims]
            snippets = vals[3]
            snippets = [clean_str(item) for sublist in snippets for item in sublist]

            all_claims += claims
            all_snippets += snippets

    all_words = [word for v in all_claims+all_snippets for word in v.split(" ")]
    counter = Counter(all_words)
    all_words = set(all_words)
    all_words = list(set([word for word in all_words if counter[word] > min_occurrence]))
    word2idx = {word: i+2 for i, word in enumerate(all_words)} # reserve 0 for potential mask and 1 for unk token
    idx2word = {word2idx[key]: key for key in word2idx}

    num_words = len(idx2word)

    glove_embedding_matrix = np.random.random((num_words+2, 300)) - 0.5
    missed = 0
    for word in word2idx:
        if word in glove_vectors:
            glove_embedding_matrix[word2idx[word]] = glove_vectors[word]
        else:
            missed += 1

    pickle.dump([glove_embedding_matrix, word2idx, idx2word], open(savename, "wb"))
    return glove_embedding_matrix, word2idx, idx2word

def train_model(model, criterion, optimizer, train_generator, val_generator, test_generator, args, other_generator, savename):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("model parameters", params)

    num_epochs = 0
    patience_counter = 0
    patience_max = 10
    best_f1 = -np.inf
    while (True):
        train_losses = []

        model.train()
        for ivals, vals in enumerate(train_generator):
            loss = train_step(optimizer, vals, model, criterion)
            train_losses.append(loss.item())

        num_epochs += 1
        print_message("TRAIN loss", np.mean(train_losses), num_epochs)

        if num_epochs % args.eval_per_epoch == 0:
            model.eval()
            with torch.no_grad():
                val_f1micro, val_f1macro, val_claimIDs, val_logits, val_labels, val_predictions = evaluate(val_generator, model)
                print_message("VALIDATION F1micro, F1macro, loss:", val_f1micro, val_f1macro, len(val_claimIDs))

            if val_f1macro > best_f1:
                with torch.no_grad():
                    test_f1micro, test_f1macro, test_claimIDs, test_logits, test_labels, test_predictions = evaluate(test_generator, model)
                    print_message("TEST F1micro, F1macro, loss:", test_f1micro, test_f1macro, len(test_claimIDs))

                    other_test_f1micro, other_test_f1macro, other_test_claimIDs, other_test_logits, other_test_labels, other_test_predictions = evaluate(other_generator, model, other_from="snes" if args.dataset == "pomt" else "pomt")
                    print_message("OTHER-TEST F1micro, F1macro, loss:", other_test_f1micro, other_test_f1macro, len(other_test_claimIDs))

                    test_remove_top_bottom = []
                    test_remove_bottom_top = []
                    other_test_remove_top_bottom = []
                    other_test_remove_bottom_top = []
                    ten = np.arange(10)
                    if args.inputtype != "CLAIM_ONLY":
                        for i in tqdm(range(10)):
                            top_is = ten[:(i+1)]
                            bottom_is = ten[-(i+1):]
                            test_remove_top_bottom.append( evaluate(test_generator, model, ignore_snippet=top_is) )
                            test_remove_bottom_top.append( evaluate(test_generator, model, ignore_snippet=bottom_is) )
                            other_test_remove_top_bottom.append(evaluate(other_generator, model, other_from="snes" if args.dataset == "pomt" else "pomt", ignore_snippet=top_is))
                            other_test_remove_bottom_top.append(evaluate(other_generator, model, other_from="snes" if args.dataset == "pomt" else "pomt", ignore_snippet=bottom_is))

                        print_message([np.around(v[1], 4) for v in test_remove_top_bottom])
                        print_message([np.around(v[1], 4) for v in test_remove_bottom_top])
                        print_message([np.around(v[1], 4) for v in other_test_remove_top_bottom])
                        print_message([np.around(v[1], 4) for v in other_test_remove_bottom_top])

                patience_counter = 0
                best_f1 = val_f1macro
                val_store = [val_f1micro, val_f1macro, val_claimIDs, val_logits, val_labels, val_predictions]
                test_store = [test_f1micro, test_f1macro, test_claimIDs, test_logits, test_labels, test_predictions, test_remove_top_bottom, test_remove_bottom_top]
                other_test_store = [other_test_f1micro, other_test_f1macro, other_test_claimIDs, other_test_logits, other_test_labels, other_test_predictions, other_test_remove_top_bottom, other_test_remove_bottom_top]
                misc_store = [args]
                total_store = [val_store, test_store, other_test_store, misc_store]
            else:
                patience_counter += 1

            print_message("PATIENCE", patience_counter, "/", patience_max)

            if patience_counter >= patience_max:
                pickle.dump(total_store, open(savename, "wb"))
                break

def run_bert(args, train_generator, val_generator, test_generator, label_weights, inputtype, label_order, savename, other_generator):
    model = MyBertModel.from_pretrained(BERT_MODEL_PATH, labelnum=len(label_order), input_type=inputtype)
    model.to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss(weight=label_weights.to(DEVICE))
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, eps=1e-8)
    optimizer.zero_grad()

    train_model(model, criterion, optimizer, train_generator, val_generator, test_generator, args, other_generator, savename)

def run_lstm(args, train_generator, val_generator, test_generator, label_weights, inputtype, label_order, savename, other_generator):
    glove_embedding_matrix, word2idx, idx2word = get_embedding_matrix([train_generator, val_generator, test_generator, other_generator], args.dataset)

    model = LSTMModel(args.lstm_hidden_dim, args.lstm_layers, args.lstm_dropout, len(label_order), word2idx, glove_embedding_matrix, input_type=inputtype)
    model.to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss(weight=label_weights.to(DEVICE))
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, eps=1e-8)
    optimizer.zero_grad()

    train_model(model, criterion, optimizer, train_generator, val_generator, test_generator, args, other_generator, savename)

def filter_snippet_for_bow(generator, ignore_snippet, inputtype):
    samples = []
    for vals in generator:
        claims = vals[1]
        labels = vals[2]
        snippets = vals[3]

        for i in range(len(snippets)):
            snippets[i][ignore_snippet] = "filler"

        for i in range(len(claims)):
            if inputtype == CLAIM_AND_EVIDENCE:
                sample = clean_str(claims[i]) + " ".join([clean_str(v) for v in snippets[i]])
            elif inputtype == CLAIM_ONLY:
                sample = clean_str(claims[i])
            elif inputtype == EVIDENCE_ONLY:
                sample = " ".join([clean_str(v) for v in snippets[i]])
            else:
                raise Exception("Unknown type", inputtype)
            samples.append(sample)
    return samples

def get_bows_labels(generators, dataset, inputtype):
    all_samples = []
    all_labels = []

    for gen in generators:
        gen_samples = []
        gen_labels = []
        for vals in gen:
            claims = vals[1]
            labels = vals[2]
            snippets = vals[3]

            for i in range(len(claims)):
                if inputtype == CLAIM_AND_EVIDENCE:
                    sample = clean_str(claims[i]) + " ".join([clean_str(v) for v in snippets[i]])
                elif inputtype == CLAIM_ONLY:
                    sample = clean_str(claims[i])
                elif inputtype == EVIDENCE_ONLY:
                    sample = " ".join([clean_str(v) for v in snippets[i]])
                else:
                    raise Exception("Unknown type", inputtype)
                gen_samples.append(sample)
                gen_labels.append(labels[i])

        all_samples.append(gen_samples)
        all_labels.append(gen_labels)

    test_remove_top_bottom = []
    test_remove_bottom_top = []
    other_test_remove_top_bottom = []
    other_test_remove_bottom_top = []
    ten = np.arange(10)
    for i in tqdm(range(10)):
        top_is = ten[:(i + 1)]
        bottom_is = ten[-(i + 1):]
        test_remove_top_bottom.append( filter_snippet_for_bow(generators[-2], top_is, inputtype) )
        test_remove_bottom_top.append( filter_snippet_for_bow(generators[-2], bottom_is, inputtype) )
        other_test_remove_top_bottom.append( filter_snippet_for_bow(generators[-1], top_is, inputtype) )
        other_test_remove_bottom_top.append( filter_snippet_for_bow(generators[-1], bottom_is, inputtype) )

    vectorizer = TfidfVectorizer(min_df=2)
    vectorizer.fit([item for sublist in all_samples for item in sublist])

    bows = [vectorizer.transform(all_samples[i]) for i in range(len(all_samples))]

    test_remove_top_bottom = [vectorizer.transform(test_remove_top_bottom[i]) for i in range(len(test_remove_top_bottom))]
    test_remove_bottom_top = [vectorizer.transform(test_remove_bottom_top[i]) for i in range(len(test_remove_bottom_top))]
    other_test_remove_top_bottom = [vectorizer.transform(other_test_remove_top_bottom[i]) for i in range(len(other_test_remove_top_bottom))]
    other_test_remove_bottom_top = [vectorizer.transform(other_test_remove_bottom_top[i]) for i in range(len(other_test_remove_bottom_top))]

    return bows, all_labels, test_remove_top_bottom, test_remove_bottom_top, other_test_remove_top_bottom, other_test_remove_bottom_top

def run_bow(args, train_generator, val_generator, test_generator, label_weights, inputtype, label_order, savename, other_test_generator):
    bows, labels, test_remove_top_bottom, test_remove_bottom_top, other_test_remove_top_bottom, other_test_remove_bottom_top = get_bows_labels([train_generator, val_generator, test_generator, other_test_generator], args.dataset, inputtype)

    train_bow, val_bow, test_bow, other_test_bow = bows[0], bows[1], bows[2], bows[3]
    train_labels, val_labels, test_labels, other_test_labels = labels[0], labels[1], labels[2], labels[3]

    label_weights = label_weights.numpy()
    weights = {i: label_weights[i] for i in range(len(label_weights))}

    param_grid = [
        {'n_estimators': [100, 500, 1000], 'min_samples_leaf': [1, 3, 5, 10], 'min_samples_split': [2, 5, 10]}
    ]

    opt = GridSearch(model=RandomForestClassifier(n_jobs=5, class_weight=weights), param_grid=param_grid, parallelize=False)
    opt.fit(train_bow, train_labels, val_bow, val_labels, scoring="f1_macro")
    exit()

    def rf_eval(model, bow, labels, other_from=None):
        preds = model.predict(bow)

        if other_from == "pomt": # other data is pomt, and model is trained on snes
            # this case is fine
            pass
        elif other_from == "snes": # other data is snes, and model is trained on pomt
            # in this case both "pants on fire!" and "false" should be considered as false
            preds[preds == 0] = 1 # 0 is "pants on fire!" and 1 is "false" for pomt.

        f1_macro = f1_score(labels, preds, average="macro")
        f1_micro = f1_score(labels, preds, average="micro")
        return f1_micro, f1_macro, labels, preds

    # val_store = [val_f1micro, val_f1macro, val_claimIDs, val_logits, val_labels, val_predictions]
    # test_store = [test_f1micro, test_f1macro, test_claimIDs, test_logits, test_labels, test_predictions,test_remove_top_bottom, test_remove_bottom_top]
    # other_test_store = [other_test_f1micro, other_test_f1macro, other_test_claimIDs, other_test_logits,
    #                     other_test_labels, other_test_predictions, other_test_remove_top_bottom,
    #                     other_test_remove_bottom_top]
    #misc_store = [args]


    val_store = rf_eval(opt, val_bow, val_labels)
    test_store = list(rf_eval(opt, test_bow, test_labels)) + [[rf_eval(opt, test_remove_top_bottom[i], test_labels) for i in range(10)],
                                                       [rf_eval(opt, test_remove_bottom_top[i], test_labels) for i in range(10)]]
    other_test_store = list(rf_eval(opt, other_test_bow, other_test_labels, other_from="snes" if args.dataset == "pomt" else "pomt")) + [[rf_eval(opt, other_test_remove_top_bottom[i], other_test_labels, other_from="snes" if args.dataset == "pomt" else "pomt") for i in range(10)],
                                                       [rf_eval(opt, other_test_remove_bottom_top[i], other_test_labels, other_from="snes" if args.dataset == "pomt" else "pomt") for i in range(10)]]
    misc_store = [opt.get_best_params()]
    total_store = [val_store, test_store, other_test_store, misc_store]

    print_message("VALIDATION", val_store[0], val_store[1])
    print_message("TEST", test_store[0], test_store[1])
    print_message("OTHER-TEST", other_test_store[0], other_test_store[1])

    print_message([np.around(v[1], 4) for v in test_store[-2]])
    print_message([np.around(v[1], 4) for v in test_store[-1]])
    print_message([np.around(v[1], 4) for v in other_test_store[-2]])
    print_message([np.around(v[1], 4) for v in other_test_store[-1]])
    print(misc_store)

    pickle.dump(total_store, open(savename, "wb"))

def filter_websites(snippets_data):
    bad_websites = ["factcheck.org", "politifact.com", "snopes.com", "fullfact.org", "factscan.ca"]
    ids = snippets_data.values[:, 0]
    remove_count = 0
    for i, id in enumerate(ids):
        with open("../../multi_fc_publicdata/snippets/" + id, "r", encoding="utf-8") as f:
            lines = f.readlines()

        links = [line.strip().split("\t")[-1] for line in lines]
        remove = [False for _ in range(10)]
        for j in range(len(links)):
            remove[j] = any([bad in links[j] for bad in bad_websites])
        remove = remove[:10]  # 1 data sample has 11 links by mistake in the dataset
        snippets_data.iloc[i, [False] + remove] = "filler"

        remove_count += np.sum(remove)
    print_message("REMOVE COUNT", remove_count)
    return snippets_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pomt", choices=["snes", "pomt"], type=str)
    parser.add_argument("--model", default="bert", choices=["bert", "lstm", "bow"], type=str)
    parser.add_argument("--eval_per_epoch", default=1, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--batchsize", default=2, type=int)
    parser.add_argument("--inputtype", default="CLAIM_AND_EVIDENCE", choices=["CLAIM_ONLY", "EVIDENCE_ONLY", "CLAIM_AND_EVIDENCE"], type=str)

    parser.add_argument("--lstm_hidden_dim", default=128, type=int)
    parser.add_argument("--lstm_layers", default=1, type=int)
    parser.add_argument("--lstm_dropout", default=0.1, type=float)

    parser.add_argument("--filter_websites", default=0, type=int)

    args = parser.parse_args()

    print_message(args)

    if args.filter_websites > 0.5:
        savename = "results/" + "-".join([str(v) for v in [args.filter_websites, args.model, args.dataset, args.inputtype, args.lr, args.batchsize]])
    else:
        savename = "results/" + "-".join([str(v) for v in [args.model, args.dataset, args.inputtype, args.lr, args.batchsize]])

    if args.model == "lstm":
        savename += "-" + "-".join([str(v) for v in [args.lstm_hidden_dim, args.lstm_layers, args.lstm_dropout]])
    savename += ".pkl"

    inputtype = INPUT_TYPE_ORDER.index(args.inputtype)
    main_data, snippets_data, label_order, splits = load_data(args.dataset)

    if args.filter_websites > 0.5:
        snippets_data = filter_websites(snippets_data)

    params = {"batch_size": args.batchsize, "shuffle": True, "num_workers": 1, "collate_fn": transformer_collate, "persistent_workers": True, "prefetch_factor":5}
    eval_params = {"batch_size": args.batchsize, "shuffle": False, "num_workers": 1, "collate_fn": transformer_collate, "persistent_workers": True, "prefetch_factor":5}

    train_generator, val_generator, test_generator, label_weights = make_generators(main_data, snippets_data, label_order, splits, [params, eval_params])

    if args.dataset == "snes":
        main_data, snippets_data, _, splits = load_data("pomt")
        if args.filter_websites > 0.5:
            snippets_data = filter_websites(snippets_data)
        main_data.iloc[main_data.iloc[:, 2] == "pants on fire!", 2] = "false"
        main_data.iloc[main_data.iloc[:, 2] == "half-true", 2] = "mixture"
        _, _, other_test_generator, _ = make_generators(main_data, snippets_data, label_order, splits, [params, eval_params], other_dataset=True)
    else:
        main_data, snippets_data, _, splits = load_data("snes")
        if args.filter_websites > 0.5:
            snippets_data = filter_websites(snippets_data)
        main_data.iloc[main_data.iloc[:, 2] == "mixture", 2] = "half-true"
        _, _, other_test_generator, _ = make_generators(main_data, snippets_data, label_order, splits, [params, eval_params], other_dataset=True)


    if args.model == "bert":
        run_bert(args, train_generator, val_generator, test_generator, label_weights, inputtype, label_order, savename, other_test_generator)
    elif args.model == "lstm":
        run_lstm(args, train_generator, val_generator, test_generator, label_weights, inputtype, label_order, savename, other_test_generator)
    elif args.model == "bow":
        run_bow(args, train_generator, val_generator, test_generator, label_weights, inputtype, label_order, savename, other_test_generator)


if __name__ == '__main__':
    main()
