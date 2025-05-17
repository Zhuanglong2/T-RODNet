from cruw import CRUW
from cruw.eval import evaluate_rod2021


data_root = r"/media/myssd/Datasets/RADAR/RADAR/mnt/disk1/CRUW/ROD2021"
submit_dir = r"/home/long/PycharmProjects/T-RODNet-main/evaluate/sub"
truth_dir = r"/home/long/PycharmProjects/T-RODNet-main/evaluate/gt"

if __name__ == '__main__':

    dataset = CRUW(data_root=data_root, sensor_config_name='sensor_config_rod2021')
    print(dataset)
    evaluate_rod2021(submit_dir, truth_dir, dataset)


