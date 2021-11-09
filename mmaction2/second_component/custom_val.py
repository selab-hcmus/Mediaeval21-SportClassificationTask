import torch
import tqdm 
from mmaction.apis import init_recognizer, inference_recognizer
import glob 
import pdb
import json

name_component_to_id_2 = ['Forehand', 'Backhand']

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
    # specify 
    config_file = '/home/nttung/Challenge/MediaevalSport/MMAction/mmaction2/second_component/config_2.py'
    checkpoint_file = '/home/nttung/Challenge/MediaevalSport/MMAction/mmaction2/second_component/checkpoint_and_log_no_bn/epoch_60.pth'
    device = 'cuda:0'
    device = torch.device(device)

    # path
    path_frames_vid_val = '/home/nttung/Challenge/MediaevalSport/MMAction/processed_frame/valid/*'
    path_gt_val_txt = '/home/nttung/Challenge/MediaevalSport/MMAction/processed_txt_second_component/val_annotate.txt'
    path_json_out = './predict_val_mmaction_second_component.json'

    # first write label path
    label_path = './label.txt'
    with open(label_path, "w") as fp:
        fp.write('\n'.join(name_component_to_id_2))

    # get gt annotation 
    with open(path_gt_val_txt, "r") as fp:
        lines = fp.readlines()
        gt_instances = [line.rstrip() for line in lines]
    
    # convert to dict for faster retrieval
    gt_dict = list2dictgt(gt_instances)

    # init model
    model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)

    gt_label_list = [] # list of interger. Each show gt label id
    predict_label_list = [] # list of interger. Each show max label id

    json_result = {}

    acc = 0
    total_sample = 0
    for vid_fr_path in glob.glob(path_frames_vid_val):
        total_sample += 1
        _, results = inference_recognizer(model, vid_fr_path, label_path, use_frames=True)
        name_instance = vid_fr_path.split("/")[-1]
        print("Process:", name_instance)


        # get gt label
        gt_label = gt_dict[name_instance]["label_id"]

        # get prediction label and score

        # get key of json for later dump
        parts = name_instance.split("_")
        json_key = "_".join([parts[0], "fr", parts[1], parts[2]])
        json_result[json_key] = {}
        json_result[json_key]["Gt"] = name_component_to_id_2[gt_label]
        json_result[json_key]["Predict"] = []

        # max result 
        max_pred = max(results, key=lambda x: x[1])
        pred_id = name_component_to_id_2.index(max_pred[0])

        if pred_id == gt_label:
            acc += 1

        # write result
        for res in results:
            json_result[json_key]["Predict"].append([res[0], str(res[1])])
        

    # calculate acc 
    print("Accuracy on val:", acc/total_sample)

    # dump result to json predict on val for later combination
    with open(path_json_out, "w") as fp:
        json.dump(json_result, fp, indent=4)