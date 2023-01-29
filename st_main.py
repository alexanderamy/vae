import torch
import streamlit as st
from codebase import utils as ut
from codebase.models.vae import VAE
from matplotlib import pyplot as plt


model = 'vae'
z = 10
run = 0
iter_max = 20000

layout = [
    ('model={:s}',  model),
    ('z={:02d}',  z),
    ('run={:04d}', run)
]

model_name = '_'.join([t.format(v) for (t, v) in layout])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = VAE(z_dim=z, name=model_name).to(device)
ut.load_model_by_name(vae, global_step=iter_max)

fig, ax = plt.subplots()

if st.button('Generate MNIST digit'):
    ax.imshow(vae.sample_x(1).detach().numpy().reshape(28,28), cmap='gray')
    ax.axis('off')
    st.pyplot(fig)
    
