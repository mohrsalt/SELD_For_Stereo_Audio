# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Reference implementation of AugMix's data augmentation method in numpy."""
import augmentations as augmentations
import numpy as np
import torch

import copy

def apply_op(op, batch_x,batch_target):
  batch_x=batch_x.cpu().detach().numpy()

  batch_x2= op(batch_x)
  batch_x2=torch.tensor(batch_x2, dtype=torch.float32)
  return batch_x2,batch_target


def augment_and_mix(batch_x,batch_target, width=3, depth=3):
  """Perform AugMix augmentations and compute mixture.

  Args:
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
"""

  batch_return=None
  batch_lab_return=None
  batch_x_aug = batch_x.clone()
  batch_target_aug = copy.deepcopy(batch_target)

  m,n,o=batch_x.shape
  c,ddim,e,f=batch_target.shape
  d = depth if depth > 0 else np.random.randint(1, 4)
  for _ in range(d):
      op = np.random.choice(augmentations.augmentations)
      batch_x_aug,batch_target_aug = apply_op(op,batch_x_aug,batch_target_aug)
    # Preprocessing commutes since all coefficients are convex
  batch_x_aug=batch_x_aug.to(device=batch_x.device)
  batch_target_aug=batch_target_aug.to(device=batch_target.device)
  batch_return=batch_x_aug
  batch_lab_return=batch_target_aug
  batch_return=batch_return.view(1,m,n,o)
  batch_lab_return=batch_lab_return.view(1,c,ddim,e,f)
  for i in range(width-1):
    batch_x_aug = batch_x.clone()
    batch_target_aug = copy.deepcopy(batch_target)
    d = depth if depth > 0 else np.random.randint(1, 4)
    for _ in range(d):
      op = np.random.choice(augmentations.augmentations)
      batch_x_aug,batch_target_aug = apply_op(op,batch_x_aug,batch_target_aug)
    # Preprocessing commutes since all coefficients are convex
    batch_x_aug=batch_x_aug.to(device=batch_x.device)
    batch_target_aug=batch_target_aug.to(device=batch_target.device)
    batch_x_aug=batch_x_aug.view(1,m,n,o)
    batch_target_aug=batch_target_aug.view(1,c,ddim,e,f)
    batch_return=torch.cat((batch_return, batch_x_aug), dim=0)
    batch_lab_return=torch.cat((batch_lab_return, batch_target_aug), dim=0)
  

  return batch_return,batch_lab_return
