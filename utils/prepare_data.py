import os
import os.path as osp
from xml.etree import ElementTree

name_component_to_id_1 = ['Serve', 'Offensive', 'Defensive']
name_component_to_id_2 = ['Forehand', 'Backhand']
name_component_to_id_3 = ['Backspin', 'Loop', 'Sidespin', 'Topspin', 'Hit', 'Flip', 'Push', 'Block']

name_component_to_id_all = ['Serve', 'Offensive', 'Defensive', 
                            'Forehand', 'Backhand',
                            'Backspin', 'Loop', 'Sidespin', 'Topspin', 'Hit', 'Flip', 'Push', 'Block']
def prepare_data(root_video_frame, root_xml_file, converted_txt_path):

    ''' 
        Create txt file for storing each instances.
        An instance is defined as:
            + id: video_id and its frame interval
            + 
    '''

    # read xml
    list_instances = []
    for xml_file in os.listdir(root_xml_file):
        print("Process video:", xml_file)
        if xml_file.endswith('.xml') is False:
            continue
        xml_path = osp.join(root_xml_file, xml_file)
        root = ElementTree.parse(xml_path).getroot()
        for instance in root:
            label_join = instance.get('move')
            label = instance.get('move').split(' ')
            video_id = xml_path.split('/')[-1][:-4]

            start, end = int(instance.get('begin')), int(instance.get('end'))
            total_fr = end - start + 1
            name_instance = "_".join([video_id, str(start), str(end)])

            # Switch between options for prepare independent data

            # label_id1 = name_component_to_id_1.index(label[0])
            # instance = " ".join([name_instance, str(total_fr), str(label_id1)])

            label_id2 = name_component_to_id_2.index(label[1])
            instance = " ".join([name_instance, str(total_fr), str(label_id2)])

            # label_id3 = name_component_to_id_3.index(label[2])
            # instance = " ".join([name_instance, str(total_fr), str(label_id3)])
            
            list_instances.append(instance)
    
    with open(converted_txt_path, "w") as fp:
        fp.write('\n'.join(list_instances))
    



if __name__ == "__main__":
    # 2021 data
    root_video_frame = '/home/nttung/Challenge/MediaevalSport/baseline/SportTaskME21/data'
    root_xml_file_train = '/home/nttung/Challenge/MediaevalSport/2021_data/data/classificationTask/train'
    root_xml_file_val = '/home/nttung/Challenge/MediaevalSport/2021_data/data/classificationTask/valid'

    converted_txt_train_path = '/home/nttung/Challenge/MediaevalSport/MMAction/processed_txt_second_component/train_annotate.txt'
    converted_txt_val_path = '/home/nttung/Challenge/MediaevalSport/MMAction/processed_txt_second_component/val_annotate.txt'
    
    # Prepare train
    prepare_data(root_video_frame, root_xml_file_train, converted_txt_train_path)

    # Prepare val
    prepare_data(root_video_frame, root_xml_file_val, converted_txt_val_path)
