import os
import json
import numpy

def convert_old_format(json_data, path_new_data):
    new_data = {}
    for each_key in json_data:
        new_data[each_key] = {"Predict": []}
        new_data[each_key]["Gt"] = json_data[each_key]["Gt"][1]

        list_comb = json_data[each_key]["Predict"]
        dict_score = {}
        for each_comb in list_comb:
            name_2nd_com = each_comb[1][0]
            score_2nd_com = each_comb[1][1]

            if name_2nd_com not in dict_score:
                dict_score[name_2nd_com] = score_2nd_com

        for name_2nd in dict_score.keys():
            new_data[each_key]["Predict"].append([name_2nd, dict_score[name_2nd]])

    with open(path_new_data, "w") as fp:
        json.dump(new_data, fp, indent=4) 

if __name__ == '__main__':
    path_old_data = "./all_component_res_val/res_all_model_12_epoch_3_8.json"
    path_new_data = "./all_component_res_val/predict_val_mmaction_second_component_converted.json"

    with open(path_old_data, 'r') as fp:
        json_data = json.load(fp)

    convert_old_format(json_data, path_new_data)