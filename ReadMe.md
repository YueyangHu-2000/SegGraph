
# SegGraph: Leveraging Graphs of SAM Segments for Few-Shot 3D Part Segmentation
> This work presents a novel framework for few-shot 3D part segmentation. Recent advances have demonstrated the significant potential of 2D foundation models for low-shot 3D part segmentation. However, it is still an open problem that how to effectively aggregate 2D knowledge from foundation models to 3D. Existing methods either ignore geometric structures for 3D feature learning or neglects the high-quality grouping clues from SAM, leading to under-segmentation and inconsistent part labels. We devise a novel SAM segment graph-based propagation method, named SegGraph, to explicitly learn geometric features encoded within SAM’s segmentation masks. Our method encodes geometric features by modeling mutual overlap and adjacency between segments while preserving intra-segment semantic consistency. We construct a segment graph, conceptually similar to an atlas, where nodes represent segments and edges capture their spatial relationships (overlap/adjacency). Each node adaptively modulates 2D foundation model features, which are then propagated via a graph neural network to learn global geometric structures. To enforce intra-segment semantic consistency, we map segment features to 3D points with a novel view-direction-weighted fusion attenuating contributions from low-quality segments. Extensive experiments on PartNet-E demonstrate that our method outperforms all competing baselines by at least 6.9% mIoU. Further analysis reveals that SegGraph achieves particularly strong performance on small components and part boundaries, demonstrating its superior geometric understanding.
## 1. Installation
### Clone Repository & Conda Env Setup
```
conda create -n SegGraph python==3.9
conda activate SegGraph
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Install Pytorch3d
```
conda install -y -c fvcore -c conda-forge iopath
pip install -U "git+https://github.com/facebookresearch/iopath.git"
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

### Install Dependencies
```
pip install -r requirements.txt
```
<!-- ### 
```
pip install scikit-learn
pip install opencv-python
pip install matplotlib
pip install open3d
pip install h5py
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install transformers
``` -->

### install SAM
```
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything/
pip install -e .
```

## 2. Data Preparation
You can download the PartNet-Mobility (PartNet-Ensembled) dataset used in the paper from [here](https://huggingface.co/datasets/minghua/PartSLIP/tree/main/).

Data preprocessing can be performed using the command below. 
```
python process/preprocess.py \
    --pc_root_path {3d data dir} \
    --process_root_path {dir to save preprocessed results}
```

### Train
```
python train.py \
  --pc_root_path {3d data dir}\
  --preprocess_root_path {dir to save preprocessed results}\
  --output_dir {output_dir}
```

### Test
```
python test.py \
  --pc_root_path {3d data dir}\
  --preprocess_root_path {dir to save preprocessed results}\
  --output_dir {output_dir}
```
