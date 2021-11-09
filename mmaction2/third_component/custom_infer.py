import torch
import tqdm 
from mmaction.apis import init_recognizer, inference_recognizer
import glob 
import pdb
import json

name_component_to_id_3 = ['Backspin', 'Loop', 'Sidespin', 'Topspin', 'Hit', 'Flip', 'Push', 'Block']

def list2dictgt(gt_instances):
    '''
        Convert list of gt instances to dictionary for instant retrieval
    '''

    gt_dict = {}

    for instance in gt_instances:
        instance = instance.split(" ")
        name_instance = instance[0]
        num_fr = int(instance[1])
        label_id = int(instance[2])

        gt_dict[name_instance] = {"total_fr": num_fr, "label_id": label_id}

    return gt_dict

if __name__ == "__main__":
    torch.cuda.empty_cache()
    # specify 
    config_file = '/home/nttung/Challenge/MediaevalSport/MMAction/mmaction2/third_component/config_3.py'
    checkpoint_file = '/home/nttung/Challenge/MediaevalSport/MMAction/mmaction2/third_component/checkpoint_and_log_no_bn/epoch_50.pth'
    device = 'cuda:0'
    device = torch.device(device)

    # path
    path_frames_vid_test = '/home/nttung/Challenge/MediaevalSport/MMAction/processed_frame/test/*'
    path_json_out = './predict_test_submit_mmaction_third_component.json'

    # first write label path
    label_path = './label.txt'
    with open(label_path, "w") as fp:
        fp.write('\n'.join(name_component_to_id_3))
    
    # init model
    model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)

    predict_label_list = [] # list of interger. Each show max label id

    json_result = {}
    
    for vid_fr_path in glob.glob(path_frames_vid_test):
        _, results = inference_recognizer(model, vid_fr_path, label_path, use_frames=True)
        name_instance = vid_fr_path.split("/")[-1]
        print("Process:", name_instance)


        # get gt label
        # gt_label = gt_dict[name_instance]["label_id"]

        # get key of json for later dump
        parts = name_instance.split("_")
        json_key = "_".join([parts[0], "fr", parts[1], parts[2]])
        json_result[json_key] = {}
        # json_result[json_key]["Gt"] = name_component_to_id_1[gt_label]
        json_result[json_key]["Predict"] = []

        # max result 
        max_pred = max(results, key=lambda x: x[1])
        pred_id = name_component_to_id_3.index(max_pred[0])

        # write result
        for res in results:
            json_result[json_key]["Predict"].append([res[0], str(res[1])])
        


    # calculate acc 
    # print("Accuracy on val:", acc/total_sample)

    # dump result to json predict on val for later combination
    with open(path_json_out, "w") as fp:
        json.dump(json_result, fp, indent=4)
