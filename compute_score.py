from utils.dataset import load_data
from utils.utils import read_rle_from_path, read_images, three_dimension_dice_score
import surface_distance
import numpy as np

def calculate_1_ratio(matrix):
    total_elements = matrix.size
    ones_count = np.count_nonzero(matrix != 0)
    ratio = ones_count / total_elements
    return ratio

def compute_3d_surface_dice_loss(label_path, pred_path,spacing_mm ):
    label = read_images(label_path)
    label = label.astype(bool)
    print('label load done')
    pred = read_rle_from_path(pred_path, label.shape[1], label.shape[2])
    pred = pred.astype(bool)
    one_ratio = calculate_1_ratio(pred)
    print('one_ratio',end = '')
    print(one_ratio)
    print('pred load done')
    three_d_dice_score = three_dimension_dice_score(pred, label)
    print('three_dimension_dice_score:',end = '')
    print(three_d_dice_score)
    surface_distances = surface_distance.compute_surface_distances(label, pred, spacing_mm)
    print('surface_distance compute done')
    score = surface_distance.compute_surface_dice_at_tolerance(surface_distances, tolerance_mm=0)
    return three_d_dice_score, score


if __name__ == "__main__":
    pred_path = 'data/predictions/prediction-pure-bce2023-12-29-12-32-27.csv'
    label_path = 'kaggle/input/blood-vessel-segmentation/train/kidney_2/labels/'
    spacing_mm = [1, 1, 1]
    three_d_dice_score, score = compute_3d_surface_dice_loss(label_path, pred_path, spacing_mm)
    print('3d_surface_dice_score:', score)