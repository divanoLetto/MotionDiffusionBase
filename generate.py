import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pathlib import Path

from src.config import read_config
from src.tools.extract_joints import extract_joints

# avoid conflic between tokenizer and rendering
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"


@hydra.main(config_path="configs", config_name="generate", version_base="1.3")
def generate(c: DictConfig):
    print("Prediction script")

    cfg = read_config(c.run_dir)
    fps = cfg.data.motion_loader.fps

    from src.text import read_texts

    print("Reading the texts")
    texts_durations = read_texts(c.input, fps)
    infos = {
        "texts_durations": texts_durations,
        "all_lengths": [x.duration for x in texts_durations],
        "all_texts": [x.text for x in texts_durations],
    }
    infos["output_lengths"] = infos["all_lengths"]
    infos["featsname"] = cfg.motion_features
    infos["guidance_weight"] = c.guidance

    print("Loading the libraries")
    import src.prepare  # noqa
    import pytorch_lightning as pl
    import numpy as np
    import torch

    ckpt_name = c.ckpt
    ckpt_path = os.path.join(c.run_dir, ckpt_name)
    print("Loading the checkpoint")
    ckpt = torch.load(ckpt_path, map_location=c.device)

    # Models
    print("Loading the models")

    # Diffusion model
    # update the folder first, in case it has been moved
    normalizer_dir = "pretrained_models/mdm-smpl_clip_smplrifke_humanml3d" if cfg.dataset=="humanml3d" else "pretrained_models/mdm-smpl_clip_smplrifke_kitml" 
    cfg.diffusion.motion_normalizer.base_dir = os.path.join(normalizer_dir, "motion_stats")
    cfg.diffusion.text_normalizer.base_dir = os.path.join(normalizer_dir, "text_stats")

    print(cfg)

    diffusion = instantiate(cfg.diffusion)
    diffusion.load_state_dict(ckpt["state_dict"])

    # Evaluation mode
    diffusion.eval()
    diffusion.to(c.device)

    # jointstype = "smpljoints"
    jointstype = "both"

    from src.tools.smpl_layer import SMPLH

    smplh = SMPLH(
        path="deps/smplh",
        jointstype=jointstype,
        input_pose_rep="axisangle",
        gender=c.gender,
    )

    from src.model.text_encoder import TextToEmb

    modelpath = cfg.data.text_encoder.modelname
    mean_pooling = cfg.data.text_encoder.mean_pooling
    text_model = TextToEmb(
        modelpath=modelpath, mean_pooling=mean_pooling, device=c.device
    )

    print("Generate the function")

    if c.seed != -1:
        pl.seed_everything(c.seed)

    # Rendering
    joints_renderer = instantiate(c.joints_renderer)
    smpl_renderer = instantiate(c.smpl_renderer)
 
    with torch.no_grad():
        tx_emb = text_model(infos["all_texts"])
        tx_emb_uncond = text_model(["" for _ in infos["all_texts"]])

        if isinstance(tx_emb, torch.Tensor):
            tx_emb = {
                "x": tx_emb[:, None],
                "length": torch.tensor([1 for _ in range(len(tx_emb))]).to(c.device),
            }
            tx_emb_uncond = {
                "x": tx_emb_uncond[:, None],
                "length": torch.tensor([1 for _ in range(len(tx_emb_uncond))]).to(c.device),
            }

        xstarts = diffusion(tx_emb, tx_emb_uncond, infos).cpu() 

    for idx, (xstart, length) in enumerate(zip(xstarts, infos["output_lengths"])):
        xstart = xstart[:length]

        extracted_output = extract_joints(
            xstart,
            infos["featsname"],
            fps=fps,
            value_from=c.value_from,
            smpl_layer=smplh,
        )

        results_dir = os.path.join(c.run_dir,f"generations")
        os.makedirs(results_dir, exist_ok=True)
        print(f"All the outputs will be saved in: {results_dir}")
        input_name =  Path(c.input).stem
        file_path = os.path.join(results_dir, input_name)

        if "smpl" in c.out_formats:
            path = file_path + "_smpl.npy"
            np.save(path, xstart)

        if "joints" in c.out_formats:
            path = file_path + "_joints.npy"
            np.save(path, extracted_output["joints"])
        
        if "vertices" in extracted_output and "vertices" in c.out_formats:
            path = file_path + "_verts.npy"
            np.save(path, extracted_output["vertices"])

        if "smpldata" in extracted_output and "smpldata" in c.out_formats:
            path = file_path + "_smpl.npz"
            np.savez(path, **extracted_output["smpldata"])

        if "videojoints" in c.out_formats:
            video_path = file_path + "_joints.mp4"
            joints_renderer(extracted_output["joints"], title="", output=video_path, canonicalize=False)

        if "vertices" in extracted_output and "videosmpl" in c.out_formats:
            print(f"SMPL rendering {idx}")
            video_path = file_path + "_smpl.mp4"
            smpl_renderer(extracted_output["vertices"], title="", output=video_path)

        if "txt" in c.out_formats:
            path = file_path + ".txt"
            with open(os.path.join(results_dir, f"{input_name}.txt"), "w") as file:
                file.write(f"Motion:\n- {texts_durations[0].text} - duration: {texts_durations[0].duration}\n")

    print("Rendering done")


if __name__ == "__main__":
    generate()