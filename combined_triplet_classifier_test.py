import json
import numpy as np
import pdb
from os import name
ref_matrix_2021 = [[1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 1, 0, 0],
                    [1, 0, 0, 0, 0, 0, 1, 1]]

name_component_to_id_1 = ['Serve', 'Offensive', 'Defensive']
name_component_to_id_2 = ['Forehand', 'Backhand']
name_component_to_id_3 = ['Backspin', 'Loop', 'Sidespin', 'Topspin', 'Hit', 'Flip', 'Push', 'Block']

# For ignoring exclusive label in 2021: Serve Backhand Loop, Serve Backhand Sidepin
list_exclusive_comb = ['Serve Backhand Loop', # Serve Backhand Loop
                'Serve Backhand Sidespin'] # Serve Backhand Sidespin

def soft_combine_json_result_test(json1, json2, json3):
    # iterate and process each key
    json_result = {}
    
    for name_instance in json1.keys():
        pred1 = json1[name_instance]["Predict"]
        pred2 = json2[name_instance]["Predict"]
        pred3 = json3[name_instance]["Predict"]

        # calculate probability score for different combinations and find max prob
        dict_score = {}
        for [lb1, score1] in pred1:
            lb1_id = name_component_to_id_1.index(lb1)
            correspondence_vector_13 = ref_matrix_2021[lb1_id]
            # first sort pred 3 again
            tmp_dict = {}
            for [lb3, score3] in pred3:
                tmp_dict[lb3] = score3

            sorted_pred3 = []
            for sorted_name in name_component_to_id_3:
                sorted_pred3.append([sorted_name, tmp_dict[sorted_name]])

            # mask out using correspondence vector first
            
            lb3_score = [float(ele[1]) for ele in sorted_pred3]
            masked_lb3_score = np.array(lb3_score) * np.array(correspondence_vector_13)
            sum_score = np.sum(masked_lb3_score)
            # normalize and pair the lb3 result
            refined_pred3 = []
            for i in range(len(masked_lb3_score)):
                refined_pred3.append([sorted_pred3[i][0], masked_lb3_score[i]/sum_score])

            for [lb2, score2] in pred2:
                for [lb3, score3] in refined_pred3:
                    triplet_predict = " ".join([lb1, lb2, lb3])
                    
                    dict_score[triplet_predict] = float(score1)*float(score2)*float(score3)

        # ignore irrelevant triplet in 2021 dataset

        for ignore_trip in list_exclusive_comb:
            dict_score[ignore_trip] = 0
        # find dict_score with max value
        max_label = max(dict_score, key=lambda label: dict_score[label])

        json_result[name_instance] = {"result": [max_label, dict_score[max_label]]}

    path_out_json = "./submission_mmaction_decompose_hybrid.json"

    with open(path_out_json, "w") as fp:
        json.dump(json_result, fp, indent=4)
         


if __name__ == '__main__':
    json_predict_1 = '/home/nttung/Challenge/MediaevalSport/MMAction/all_component_res_test/predict_test_submit_mmaction_first_component.json'
    json_predict_2 = '/home/nttung/Challenge/MediaevalSport/MMAction/all_component_res_test/predict_test_submit_mmaction_second_component.json'
    json_predict_3 = '/home/nttung/Challenge/MediaevalSport/MMAction/all_component_res_test/predict_test_submit_mmaction_third_component.json'


    # read json
    with open(json_predict_1, "r") as fp:
        json1 = json.load(fp)

    with open(json_predict_2, "r") as fp:
        json2 = json.load(fp)

    with open(json_predict_3, "r") as fp:
        json3 = json.load(fp)

    soft_combine_json_result_test(json1, json2, json3)
    
