from mmcv.runner import set_random_seed
from mmcv import Config
from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.apis import train_model

import mmcv
import os.path as osp

name_component_to_id_all = ['Serve', 'Offensive', 'Defensive', 
                            'Forehand', 'Backhand',
                            'Backspin', 'Loop', 'Sidespin', 'Topspin', 'Hit', 'Flip', 'Push', 'Block']
name_component_to_id_1 = ['Serve', 'Offensive', 'Defensive']


def get_cfg():
    

    # path = '/home/nttung/Challenge/MediaevalSport/MMAction/mmaction2/second_component/config_2.py'
    path = '/home/nttung/Challenge/MediaevalSport/MMAction/mmaction2/configs/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb.py'
    # path = '/home/nttung/Challenge/MediaevalSport/MMAction/mmaction2/configs/recognition/csn/ircsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb.py'
    cfg = Config.fromfile(path)
    # Modify dataset type and path
    cfg.dataset_type = 'RawframeDataset'
    cfg.data_root = '/home/nttung/Challenge/MediaevalSport/MMAction/processed_frame/train'
    cfg.data_root_val = '/home/nttung/Challenge/MediaevalSport/MMAction/processed_frame/valid'
    cfg.ann_file_train = '/home/nttung/Challenge/MediaevalSport/MMAction/processed_txt_first_component/train_annotate.txt'
    cfg.ann_file_val = '/home/nttung/Challenge/MediaevalSport/MMAction/processed_txt_first_component/val_annotate.txt'
    cfg.ann_file_test = '/home/nttung/Challenge/MediaevalSport/MMAction/processed_txt_first_component/val_annotate.txt'

    cfg.data.test.type = 'RawframeDataset'
    cfg.data.test.ann_file = '/home/nttung/Challenge/MediaevalSport/MMAction/processed_txt_first_component/val_annotate.txt'
    cfg.data.test.data_prefix = '/home/nttung/Challenge/MediaevalSport/MMAction/processed_frame/valid'

    cfg.data.train.type = 'RawframeDataset'
    cfg.data.train.ann_file =  '/home/nttung/Challenge/MediaevalSport/MMAction/processed_txt_first_component/train_annotate.txt'
    cfg.data.train.data_prefix = '/home/nttung/Challenge/MediaevalSport/MMAction/processed_frame/train'

    cfg.data.val.type = 'RawframeDataset'
    cfg.data.val.ann_file = '/home/nttung/Challenge/MediaevalSport/MMAction/processed_txt_first_component/val_annotate.txt'
    cfg.data.val.data_prefix = '/home/nttung/Challenge/MediaevalSport/MMAction/processed_frame/valid'

    # cfg.setdefault('omnisource', False)
    # Modify num classes of the model in cls_head
    cfg.model.cls_head.num_classes = len(name_component_to_id_1)
    # We can use the pre-trained TSN model
    # cfg.load_from = '/content/drive/My Drive/0TruongHai/Sport_Multi/epoch_100.pth'
    cfg.load_from = '/home/nttung/Challenge/MediaevalSport/MMAction/mmaction2/checkpoints/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb_20200812-9037a758.pth'
    # Set up working dir to save files and logs.
    cfg.work_dir = './checkpoint_and_log_bn'
    cfg.total_epochs = 100
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    return cfg

if __name__ == "__main__":
    # Get cfg
    cfg = get_cfg()
    print(cfg.pretty_text)
    print("Training first component")
    # Build the dataset

    datasets = [build_dataset(cfg.data.train)]

    # Build the recognizer
    model = build_model(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_model(model, datasets, cfg, distributed=False, validate=True)