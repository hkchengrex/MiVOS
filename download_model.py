import os
import gdown
import urllib.request


os.makedirs('saves', exist_ok=True)
print('Downloading propagation model...')
gdown.download('https://drive.google.com/uc?id=19dfbVDndFkboGLHESi8DGtuxF1B21Nm8', output='saves/propagation_model.pth', quiet=False)

print('Downloading fusion model...')
gdown.download('https://drive.google.com/uc?id=1Lc1lI5-ix4WsCRdipACXgvS3G-o0lMoz', output='saves/fusion.pth', quiet=False)

print('Downloading s2m model...')
gdown.download('https://drive.google.com/uc?id=1HKwklVey3P2jmmdmrACFlkXtcvNxbKMM', output='saves/s2m.pth', quiet=False)

print('Downloading fbrs model...')
urllib.request.urlretrieve('https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/resnet50_dh128_lvis.pth', 'saves/fbrs.pth')

print('Done.')