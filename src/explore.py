import torch
from red_motion.models.red_motion import RedMotionCrossFusion

torch.cuda.empty_cache()


model = RedMotionCrossFusion(
    dim_road_env_encoder=128,
    dim_road_env_attn_window=16,
    dim_ego_trajectory_encoder=128,
    num_trajectory_proposals=6,
    prediction_horizon=50,
    learning_rate=0,
    batch_size=1,
    prediction_subsampling_rate=5,
    mode="pre-training",
    reduction_feature_aggregation="learned",
)
model.load_state_dict(
    torch.load("../../cross-fusion-red_motion-2023-09-27T13-04-03.pt")
)

model.to("cuda:2")
model.eval()


from torch.utils.data import DataLoader
from red_motion.data_utils.dataset_modules import WaymoRoadEnvGraphDataset

loader = DataLoader(
    WaymoRoadEnvGraphDataset(directory="red_motion/data_utils/demo_files"),
    batch_size=1,
    shuffle=False,
)


import numpy as np
from glob import glob

from tqdm.auto import tqdm
from red_motion.models.red_motion import red_motion_inference

samples = sorted(glob("red_motion/data_utils/demo_files/*.npz"))
data = [np.load(sample) for sample in samples]
predictions = []

with torch.no_grad():
    for idx, batch in tqdm(enumerate(loader)):
        logits_np, confidences_np, is_available_np, y_np = red_motion_inference(
            model, batch, return_all_numpy=True
        )
        predictions.append(
            {
                "logits": logits_np,
                "confidences": confidences_np,
                "is_available": is_available_np,
                "ground_truth": y_np,
            }
        )

from red_motion.data_utils.visualize import plot_marginal_predictions_3d

idx = 1

plot_marginal_predictions_3d(
    data[idx]["vector_data"],
    predictions=predictions[idx]["logits"],
    x_range=(-20, 50),
    y_range=(-20, 50),
    confidences=predictions[idx]["confidences"],
    is_available=data[idx]["future_val_marginal"],
    gt_marginal=data[idx]["gt_marginal"],
)

idx = 4
plot_marginal_predictions_3d(
    data[idx]["vector_data"],
    predictions=predictions[idx]["logits"],
    x_range=(-20, 50),
    y_range=(-20, 50),
    confidences=predictions[idx]["confidences"],
    is_available=data[idx]["future_val_marginal"],
    gt_marginal=data[idx]["gt_marginal"],
)

idx = 5
plot_marginal_predictions_3d(
    data[idx]["vector_data"],
    predictions=predictions[idx]["logits"],
    x_range=(-20, 50),
    y_range=(-20, 50),
    confidences=predictions[idx]["confidences"],
    is_available=data[idx]["future_val_marginal"],
    gt_marginal=data[idx]["gt_marginal"],
)
