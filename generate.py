#!/usr/bin/env python
# coding: utf-8

"""
Created on Mon Jul 14 17:09:12 2025

@author: LvXiang
"""
import os
import yaml
import numpy as np
import torch
from src.diffusion_src.models.gaussian_diffusion import GaussianDiffusion
from src.diffusion_src.models.scAttNet import SingleCellAN

def sample_cells_chunked(model, cd_dict, gaussian_diffusion, device,
                         num_cells_total, cell_num_per_sample,
                         feature_num, output_dir):
    """
    Generate synthetic single–cell measurements in chunks for each donor,
    saving each donor’s array to a .npy file.
    
    Args:
        model: Trained SingleCellAN model.
        cd_dict: Mapping donor_id -> precomputed EHR embedding tensor.
        gaussian_diffusion: GaussianDiffusion sampler.
        loader: DataLoader over a subset of the dataset (to get donor IDs).
        device: torch.device.
        num_cells_total: Total number of cells to generate per donor.
        cell_num_per_sample: Cells generated per call to diffusion.sample().
        feature_num: Dimensionality of each cell feature vector.
        output_dir: Directory in which to save .npy files.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    seen = set()
    print("Genaration process Begin")
    with torch.no_grad():
        for did in cd_dict.keys():
            if did in seen: continue
            seen.add(did)
            ehr_emb = cd_dict[did].unsqueeze(1).to(device)
            mask    = torch.ones(ehr_emb.size(0), 1, dtype=torch.bool, device=device)
            total = []
            while len(total) < num_cells_total:
                gen = gaussian_diffusion.sample(
                    model=model,
                    batch_size=1,
                    cell_num=cell_num_per_sample,
                    dims=feature_num,
                    cd=(ehr_emb, mask)
                )
                total.extend(gen.squeeze(0).cpu().tolist())
            arr = np.array(total[:num_cells_total], dtype=np.float32)
            np.save(os.path.join(output_dir, f"{did}.npy"), arr)
            print(f"generated {did}")
        print("Genaration process Finish!")

def load_cfg(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def strip_ddp_prefix(state_dict: dict) -> dict:
    """
    Remove the 'module.' prefix inserted by DDP from each key, so that
    the weights can be loaded into a plain nn.Module.
    Args:
        state_dict: The raw state_dict, possibly with 'module.' prefixes.
    Returns:
        A new state_dict without the 'module.' prefixes.
    """
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v
    return new_state


def main():
    """
    Evaluate the diffusion model by loading the best checkpoint and
    generating per-donor single-cell arrays.
    """
    cd_dict = torch.load("data/cd_dict.pt")
    cd_dict_v = [v for k,v in cd_dict.items()]
    emb_dim= cd_dict_v[0].shape[1]
    cfg = load_cfg(os.path.join("configs", "diffusion.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    diff_model = SingleCellAN(
        feature_dims=cfg["model"]["feature_dims"],
        EHR_embdims=emb_dim,
        model_dims=cfg["model"]["model_dims"],
        dims_mult=tuple(cfg["model"]["dims_mult"]),
        num_res_blocks=cfg["model"]["num_res_blocks"],
        attention_resolutions=tuple(cfg["model"]["attention_resolutions"]),
        dropout=cfg["model"]["dropout"],
        dropoutAtt=cfg["model"]["dropout_att"],
        num_heads=cfg["model"]["num_heads"],
    ).to(device)
    raw = torch.load(os.path.join("checkpoints/diffusion_ckpt", "best_diff_model.pth"),
                     map_location=device)
    diff_model.load_state_dict(strip_ddp_prefix(raw))
    diff_model.eval()
    gd = GaussianDiffusion(timesteps=cfg["diffusion"]["timesteps"])
    sample_cells_chunked(
        model=diff_model,
        cd_dict=cd_dict,
        gaussian_diffusion=gd,
        device=device,
        num_cells_total=cfg["evaluation"]["num_cells_total"],
        cell_num_per_sample=cfg["evaluation"]["cell_num_per_sample"],
        feature_num=cfg["model"]["feature_dims"],
        output_dir="./sample_cells"
    )
    
if __name__ == "__main__":
    main()

