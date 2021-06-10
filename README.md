# MiVOS - STCN backbone

[Ho Kei Cheng](https://hkchengrex.github.io/), Yu-Wing Tai, Chi-Keung Tang

CVPR 2021

### We replace the propagatron backbone with [SCTN](https://github.com/hkchengrex/STCN) in this branch. It is better and faster! Only the essentials are covered here. See the master branch for the rest.

## Quick start

### GUI

1. `python download_model.py` to get all the required models.
2. `python interactive_gui.py --video <path to video>` or `python interactive_gui.py --images <path to a folder of images>`. A video has been prepared for you at `examples/example.mp4`.
3. If you need to label more than one object, additionally specify `--num_objects <number_of_objects>`. See all the argument options with `python interactive_gui.py --help`.
4. There are instructions in the GUI. You can also watch the [demo videos](https://hkchengrex.github.io/MiVOS/video.html#partb) for some ideas.

### DAVIS Interactive VOS

See `eval_interactive_davis.py`. If you have downloaded the datasets and pretrained models using our script, you only need to specify the output path, i.e., `python eval_interactive_davis.py --output [somewhere]`.

### DAVIS/YouTube Semi-supervised VOS

Go to this repo: [STCN](https://github.com/hkchengrex/STCN).

## Main Results

### DAVIS Interactive Track

| Model | AUC-J&F | J&F @ 60s |
| --- |:--:|:---:|
| MiVOS + STM best model | 87.9 | 88.5 |
| MiVOS + STCN | 88.4 | 88.8 |

(Performance mostly bottlenecked by interaction. Speed-wise it improves quite a bit -- try the GUI.)

## Pretrained models

`python download_model.py` should get you all the models that you need. (`pip install gdown` required.)

Google Drive:

[[stcn.pth]](https://drive.google.com/file/d/1mRrE0uCI2ktdWlUgapJI_KmgeIiF2eOm/view?usp=sharing)
[[fusion_stcn.pth]](https://drive.google.com/file/d/1MAbWHrOjlze9vPQdW-HxMnvjPpaZlfLv/view?usp=sharing)

OneDrive:

[[stcn.pth]](https://uillinoisedu-my.sharepoint.com/:u:/g/personal/hokeikc2_illinois_edu/Eav35v3GZIZFiq6dv9BM8n0BHtR1hD7QU9tcxH7hylG3dA?e=ZQmPJh)
[[fusion_stcn.pth]](https://uillinoisedu-my.sharepoint.com/:u:/g/personal/hokeikc2_illinois_edu/Eflt9urRY2VBgXMKzk0Or8QBFXdV-CVUSDhjuOa9zJJ0Gw?e=jLfHVN)

## Training

Use the same commands as in master.

## Citation

Please cite our papers if you find this repo useful!

```bibtex
@inproceedings{cheng2021stcn,
  title={Rethinking Space-Time Networks with Improved Memory Coverage for Efficient Video Object Segmentation},
  author={Cheng, Ho Kei and Tai, Yu-Wing and Tang, Chi-Keung},
  booktitle={arXiv:2106.05210},
  year={2021}
}

@inproceedings{cheng2021mivos,
  title={Modular Interactive Video Object Segmentation: Interaction-to-Mask, Propagation and Difference-Aware Fusion},
  author={Cheng, Ho Kei and Tai, Yu-Wing and Tang, Chi-Keung},
  booktitle={CVPR},
  year={2021}
}
```

Contact: <hkchengrex@gmail.com>
