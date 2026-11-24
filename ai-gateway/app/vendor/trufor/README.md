# TruFor (vendored)

Source: https://github.com/grip-unina/TruFor, `test_docker/src/models`, commit `ae54475`.
Copyright (c) 2023 GRIP-UNINA. **Nonprofit use only.** See `LICENSE.txt` and `LICENSE_CMX.txt`.
Cite the TruFor paper (Guillaro et al., CVPR 2023) whenever results from it are made public.

Local changes, needed only to run it inside the gateway package:
- `models/cmx/builder_np_conf.py`: `make_net` is imported relatively.
- `models/cmx/encoders/dual_segformer.py` and `models/cmx/net_utils.py`: the `timm` helpers come from the
  new file `models/cmx/_timm_compat.py`.
- Empty `__init__.py` files were added so the folders are regular packages.

The weights are not stored here. Download `TruFor_weights.zip` (see the repository README) and put
`trufor.pth.tar` in `models/image-trufor/`.
