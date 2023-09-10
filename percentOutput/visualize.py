import numpy as np

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

NUM_PERCENT = 9

CTG = 2232.875
W = 0
AIR = -1000

# plotly view
fig = make_subplots(3, 3, horizontal_spacing=0.01, vertical_spacing=0.01)
# read img
for i in range(NUM_PERCENT):
    hovertemplate = "x: %{x} <br> y: %{y} <br> z: %{z} <br> ct: %{customdata[0]:.4f} <br> percent: %{customdata[1]:.4f},  %{customdata[2]:.4f}, %{customdata[3]:.4f}"

    img = np.load(f'./percentOutput/image_np/img_{i+10}_0.npy')
    img = img.reshape(img.shape[-2], img.shape[-1])

    ct = ((img + 1) / 2) * (CTG - AIR) + AIR

    percent = np.load(f'./percentOutput/percent_np/percent_{i+10}_0.npy').reshape(
        img.shape[-2], img.shape[1], 3)

    fig.add_trace(go.Heatmap(z=img,
                             customdata=np.dstack((ct, percent)),
                             hovertemplate=hovertemplate),
                  row=int(i / 3) + 1,
                  col=i % 3 + 1)

# fig.show()

fig.write_html("/home/jeffbla/pytorch-CycleGAN-and-pix2pix/percentOutput/visual.html")