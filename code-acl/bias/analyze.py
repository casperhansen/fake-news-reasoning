import pickle
import numpy as np
import pandas as pd
import glob
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

inputtypes = ["CLAIM_ONLY", "EVIDENCE_ONLY", "CLAIM_AND_EVIDENCE"]
inputtypes_dic = {"CLAIM_ONLY": "Claim", "EVIDENCE_ONLY": "Evidence", "CLAIM_AND_EVIDENCE": "Claim+Evidence"}
datasets = ["snes", "pomt"]
datasets_dic = {"snes": "Snopes", "pomt": "PolitiFact"}
methods = ["bow", "lstm", "bert"]
methods_dic = {"bow": "RF", "lstm": "LSTM", "bert": "BERT"}

def process_content(content):
    val_store = content[0]
    test_store = content[1]
    other_test_store = content[2]
    misc = content[3]

    test_remove_top_bottom = content[1][-2]
    test_remove_bottom_top = content[1][-1]
    other_test_remove_top_bottom = content[2][-2]
    other_test_remove_bottom_top = content[2][-1]

    val_f1_micro = val_store[0]
    val_f1_macro = val_store[1]

    test_f1_micro = test_store[0]
    test_f1_macro = test_store[1]

    other_test_f1_micro = other_test_store[0]
    other_test_f1_macro = other_test_store[1]

    test_remove_top_bottom_f1_micro = [v[0] for v in test_remove_top_bottom]
    test_remove_top_bottom_f1_macro = [v[1] for v in test_remove_top_bottom]

    test_remove_bottom_top_f1_micro = [v[0] for v in test_remove_bottom_top]
    test_remove_bottom_top_f1_macro = [v[1] for v in test_remove_bottom_top]

    other_test_remove_top_bottom_f1_micro = [v[0] for v in other_test_remove_top_bottom]
    other_test_remove_top_bottom_macro = [v[1] for v in other_test_remove_top_bottom]

    other_test_remove_bottom_top_f1_micro = [v[0] for v in other_test_remove_bottom_top]
    other_test_remove_bottom_top_f1_macro = [v[1] for v in other_test_remove_bottom_top]

    return val_f1_micro, val_f1_macro, test_f1_micro, test_f1_macro, other_test_f1_micro, other_test_f1_macro, \
           test_remove_top_bottom_f1_micro, test_remove_top_bottom_f1_macro, \
           test_remove_bottom_top_f1_micro, test_remove_bottom_top_f1_macro, \
           other_test_remove_top_bottom_f1_micro, other_test_remove_top_bottom_macro, \
           other_test_remove_bottom_top_f1_micro, other_test_remove_bottom_top_f1_macro

def get_results(dataset, inputtype, model, other=None):
    path = "results/"

    filenames = glob.glob(path + "-".join([str(v) for v in [model, dataset, inputtype]]) + "*.pkl")
    if other is not None:
        filenames = glob.glob(path + "-".join([str(v) for v in [other, model, dataset, inputtype]]) + "*.pkl")

    best_val_f1_macro = -np.inf
    best_filename = None
    best_misc = None
    for filename in filenames:
        content = pickle.load(open(filename, "rb"))
        val_f1_macro = (content[0][1]+content[0][0])/2

        if val_f1_macro > best_val_f1_macro:
            best_filename = filename
            best_val_f1_macro = val_f1_macro
            best_misc = content[-1]

    content = pickle.load(open(best_filename, "rb"))
    print(dataset, inputtype, model,)
    print(best_misc)
    print("--")
    return process_content(content)

def get_test_f1_micro(vals):
    return vals[2]

def get_test_f1_macro(vals):
    return vals[3]

def get_other_test_f1_micro(vals):
    return vals[4]

def get_other_test_f1_macro(vals):
    return vals[5]

def get_test_remove_top_bottom_f1_micro(vals):
    return vals[6]

def get_test_remove_top_bottom_f1_macro(vals):
    return vals[7]

def get_test_remove_bottom_top_f1_micro(vals):
    return vals[8]

def get_test_remove_bottom_top_f1_macro(vals):
    return vals[9]

def my_round(val):
    return "{:.3f}".format(val)

#inputtypes = ["CLAIM_ONLY", "EVIDENCE_ONLY", "CLAIM_AND_EVIDENCE"]
def plot_snippets(method, dataset, resdic):
    evid = resdic[method][dataset]["EVIDENCE_ONLY"]
    claim_evid = resdic[method][dataset]["CLAIM_AND_EVIDENCE"]

    evid_original = [get_test_f1_macro(evid)]
    claim_evid_original = [get_test_f1_macro(claim_evid)]

    evid_top = evid_original + get_test_remove_top_bottom_f1_macro(evid)
    evid_bottom = evid_original + get_test_remove_bottom_top_f1_macro(evid)

    claim_evid_top = claim_evid_original + get_test_remove_top_bottom_f1_macro(claim_evid)
    claim_evid_bottom = claim_evid_original + get_test_remove_bottom_top_f1_macro(claim_evid)

    plt.figure(figsize=(5, 3.5))
    xs = np.arange(11)
    plt.plot(xs, evid_top, "--*b", label="Evid: remove from top")
    plt.plot(xs, evid_bottom, "-*b", label="Evid: remove from bottom")

    plt.plot(xs, claim_evid_top, "--+r", label="Claim+Evid: remove from top")
    plt.plot(xs, claim_evid_bottom, "-+r", label="Claim+Evid: remove from bottom")

    plt.title(datasets_dic[dataset] + ": " + methods_dic[method])
    plt.legend()
    plt.xlabel("#snippets removed")
    plt.ylabel("F1 macro")
    plt.xticks(xs)
    plt.tight_layout()
    plt.savefig("figs/" + datasets_dic[dataset] + "_" + methods_dic[method] + ".pdf")

def print_table(method, resdic): # [method][dataset][t]
    print(method)
    for inputtype in inputtypes:
        line = inputtypes_dic[inputtype]
        for dataset in datasets:
            for fun in [get_test_f1_micro, get_test_f1_macro, get_other_test_f1_micro, get_other_test_f1_macro]:
                line = line + " & " + my_round(fun(resdic[method][dataset][inputtype]))
        line += " \\\\"
        print(line)

def main():
    resdic = {}
    for method in methods:
        resdic[method] = {}
        for dataset in datasets:
            resdic[method][dataset] = {}
            for t in inputtypes:
                resdic[method][dataset][t] = get_results(dataset, t, method)


    print_table("bow", resdic)
    print_table("lstm", resdic)
    print_table("bert", resdic)

    plot_snippets("bert", "snes", resdic)
    plot_snippets("bert", "pomt", resdic)

    plot_snippets("lstm", "snes", resdic)
    plot_snippets("lstm", "pomt", resdic)

    plot_snippets("bow", "snes", resdic)
    plot_snippets("bow", "pomt", resdic)

    plt.show()

if __name__ == '__main__':
    main()