import os.path as osp 
import os
from moviepy.editor import *
from tqdm import tqdm
from xml.etree import ElementTree
import pdb

if __name__ == "__main__":
    count = 0
    data_path = "/home/nttung/Challenge/MediaevalSport/2021_data/data/classificationTask"
    processed_video_path = "/home/nttung/Challenge/MediaevalSport/MMAction/processed_video"

    for folder in os.listdir(data_path):
        if folder == "train":
            continue
        for xml_file in os.listdir(os.path.join(data_path, folder)):
            if xml_file.endswith('.xml') is False:
                continue

            video_name = xml_file.replace(".xml", ".mp4")
            clip = VideoFileClip("/home/nttung/Challenge/MediaevalSport/2021_data/data/videos/{}".format(video_name))

            xml_path = osp.join(data_path, folder, xml_file)
            root = ElementTree.parse(xml_path).getroot()
            
            for instance in root:
                label_join = instance.get('move')
                label = instance.get('move').split(' ')
                sequence = [int(instance.get('begin')), int(instance.get('end'))]

                outclip = clip.subclip(sequence[0]/clip.fps, sequence[1]/clip.fps)
                # pdb.set_trace()
                outclip.write_videofile("{}/{}/{}_{}_{}.mp4".format(processed_video_path, folder, video_name[:-4], sequence[0], sequence[1]))
                count += 1
                print("At: ", count)

    print("Total number of videos: ", count)


