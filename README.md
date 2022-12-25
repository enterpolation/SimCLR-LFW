# Supervised contrastive learning on LFW dataset
`Dataset`: [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/).


## Review
Our  goal is to learn such an embedding space in which similar sample pairs stay 
close to each other while dissimilar ones are far apart.

### Model structure
SimCLR ([Chen et al, 2020](https://arxiv.org/pdf/2002.05709.pdf)) proposed a simple framework for 
contrastive learning of visual representations. It learns representations for visual inputs 
by maximizing agreement between differently augmented views of the same sample via a contrastive 
loss in the latent space.

<p align="center">
    <img src=images/Model.jpg width=50% />
</p>

This category of approaches produce two noise versions of one anchor image and aim to 
learn representation such that these two augmented samples share the same embedding. 
The algorithm is following:

**1.** Randomly sample a minibatch of $N$ samples and each sample is applied with 
two different data augmentation operations, resulting in $2N$ augmented samples in total.
```math
\tilde{\mathbf{x}}_i = t(\mathbf{x}),\quad\tilde{\mathbf{x}}_j = 
t'(\mathbf{x}),\quad t, t' \sim \mathcal{T}
```
where two separate data augmentation operators, $t$ and $t'$, are sampled from 
the same family of augmentations $\mathcal{T}$. Data augmentation includes: 
- random crop;
- random flip;
- random rotation;
- color jitter;
- gaussian blur.

**2.** Given one positive pair, other $2(N-1)$ data points 
are treated as negative samples. The representation is produced by a base encoder $f(.)$:
```math
\mathbf{h}_i = f(\tilde{\mathbf{x}}_i),\quad \mathbf{h}_j = f(\tilde{\mathbf{x}}_j)
```

**3.** The contrastive learning loss is defined using cosine similarity $\text{sim}(.,.)$. 
Note that the loss operates on an extra projection layer of the representation $g(.)$ rather than 
on the representation space directly. But only the representation $\mathbf{h}$ is used 
for downstream tasks.
```math
\mathbf{z}_i = g(\mathbf{h}_i),\quad
\mathbf{z}_j = g(\mathbf{h}_j)
```
Since we have labels for the dataset, we will be using supervised contrastive loss (`SupConLoss`):
```math
\mathcal{L}_\text{supcon} = 
- \sum_{i=1}^{2n} \frac{1}{2 \vert N_i \vert - 1} 
\sum_{j \in N(y_i), j \neq i} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_j / \tau)}
{\sum_{k \in I, k \neq i}\exp({\mathbf{z}_i \cdot \mathbf{z}_k / \tau})}
```
`SimCLR` needs a large batch size to incorporate enough negative samples to achieve good performance.
<p align="center">
    <img src=images/Algorithm.jpg width=50% />
</p>


## Quick start
All actions should be done from the inside `./` directory.

### Setup
You can set all the model parameters in the `./source/config.py` file:
```python
import torch


ORIGINAL_SIZE = 255  # original image size
IMAGE_SIZE = 64  # augmented image size
BATCH_SIZE = 128
LEARNING_RATE = 0.1
NUM_EPOCH = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### Train
```
python source/train.py \
--model_path models/model.pth \
--data_path data/
```

### Predict
```
python source/predict.py \
--model_path models/model.pth \
--image_path_1 images/Aaron_Peirsol_0001.jpg \
--image_path_2 images/Aaron_Peirsol_0002.jpg
```

The output is a cosine similarity between 2 given images.


### Visualization
```
python source/visualize.py \
--data_path data/ \
--model_path models/model.pth \
--plot_path images/tsne.jpg \
--k_classes 3
```

Result:
<p align="center">
    <img src=images/tsne.jpg width=50% />
</p>