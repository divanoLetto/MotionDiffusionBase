import numpy as np
import torch

from src.tools.smpl_layer import SMPLH
from src.tools.extract_joints import extract_joints

from TMR.mtt.load_tmr_model import load_tmr_model_easy
from TMR.src.guofeats import joints_to_guofeats
from TMR.src.model.tmr import get_sim_matrix

# carico  il modello TMR
tmr_forward = load_tmr_model_easy(device="cpu", dataset="humanml3d") 


def calc_eval_stats(x_rec, x_gt, texts_gt):
    """
    Calculate Motion2Motion (m2m) and the Motion2Text (m2t) between the recostructed motion, the gt motion and the gt text.
    """

    text_latents_gt = tmr_forward(texts_gt) #  tensor(N, 256) 
    smplh = SMPLH(
        path="deps/smplh",
        jointstype='both',
        input_pose_rep="axisangle",
        gender='male',
    )
    x_gt_output = extract_joints(
        x_gt,
        'smplrifke',
        fps=20,
        value_from='smpl',
        smpl_layer=smplh,
    )
    x_rec_output = extract_joints(
        x_rec,
        'smplrifke',
        fps=20,
        value_from='smpl',
        smpl_layer=smplh,
    )
    x_gt_joints = x_gt_output["joints"]
    x_rec_joints = x_rec_output["joints"]
    x_gt_guofeats = joints_to_guofeats(x_gt_joints) 
    x_rec_guofeats = joints_to_guofeats(x_rec_joints)

    motion_latents_gt = tmr_forward([x_gt_guofeats])  # tensor(N, 256)
    motion_latents = tmr_forward([x_rec_guofeats])  # tensor(N, 256)

    sim_matrix_m2t = get_sim_matrix(motion_latents, text_latents_gt).numpy()
    sim_matrix_m2m = get_sim_matrix(motion_latents, motion_latents_gt).numpy()

    m2m, m2t = sim_matrix_m2t[0,0], sim_matrix_m2m[0,0] 

    return m2m, m2t

### Esempio di calcolo di M2T e M2M con bs=1

# Si caricano due tensori (1, #frames, 205), rappresentano il motion recostruito e il motion ground truth
x_1 = np.load("pretrained_models/mdm-smpl_clip_smplrifke_humanml3d/generations/example_smpl.npy")
x_2 = np.load("pretrained_models/mdm-smpl_clip_smplrifke_humanml3d/generations/example_smpl.npy")
x_rec = torch.tensor(x_1)
x_gt =  torch.tensor(x_2)

# Si caricano i testi relativi alle gt motions
texts_gt = ["walking"]

# Si calcolano le metriche
m2m, m2t = calc_eval_stats(x_rec, x_gt, texts_gt)

print(f"m2m: {m2m}, m2t: {m2t}")