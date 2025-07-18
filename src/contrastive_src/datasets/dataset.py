# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 16:22:33 2025

@author: LvXiang
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Tuple

class EHRSingleCellDataset(Dataset):
    """
    Pairs each patient’s EHR vector with one of their
    CyTOF single-cell feature matrices and a numeric label.
    """
    def __init__(self, ehr_csv: str, sc_csv: str):
        # 1) Load EHR CSV → donor_ID → vector
        ehr_df = pd.read_csv(ehr_csv)
        self.ehr_dict = {
            row['donor_ID']: row.drop('donor_ID').to_numpy(dtype=np.float32)
            for _, row in ehr_df.iterrows()
        }
        donors = sorted(self.ehr_dict.keys())
        self.donor2label = {d: i for i, d in enumerate(donors)}

        # 2) Load single‐cell CSV and group by 'sample'
        sc_df = pd.read_csv(sc_csv)
        self.sc_dict = defaultdict(dict)
        for sample_id, grp in sc_df.groupby('sample'):
            parts = sample_id.split('-')
            donor = '-'.join(parts[:2])
            grp_idx = int(parts[2]) if len(parts) > 2 else 0
            # drop only 'sample' column, convert to float32 array
            arr = grp.drop(columns=['sample']).to_numpy(dtype=np.float32)
            self.sc_dict[donor][grp_idx] = arr

        # 3) Build flat list of (donor, grp_idx, label)
        self.samples: List[Tuple[str,int,int]] = []
        for donor, groups in self.sc_dict.items():
            if donor not in self.ehr_dict:
                continue
            lbl = self.donor2label[donor]
            for idx in groups:
                self.samples.append((donor, idx, lbl))

    def __len__(self) -> int:
        """Total number of (donor, group) pairs."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Returns:
          sc_tensor    (cells, sc_features)
          ehr_tensor   (ehr_features,)
          label_tensor scalar long
          donor_id     string
        """
        donor, grp_idx, label = self.samples[idx]
        sc_arr  = self.sc_dict[donor][grp_idx]
        ehr_arr = self.ehr_dict[donor]
        return (
            torch.from_numpy(sc_arr),
            torch.from_numpy(ehr_arr),
            torch.tensor(label, dtype=torch.long),
            donor
        )


