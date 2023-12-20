from utils.utils import read_rle_from_path, three_dimension_dice_score
from utils.dataset import load_data

target_path = 'kaggle/input/blood-vessel-segmentation/train/kidney_2'
target = load_data(target_path, "/labels/").numpy()

pred_path = 'data/predictions/prediction2023-12-1216-52-47.csv'
pred = read_rle_from_path(pred_path, target.shape[1], target.shape[2])

score = three_dimension_dice_score(pred, target)
print('the dice score is', score)