# Collaborative Filtering on Bipartite Graphs using Graph Convolutional Networks

A PyTorch implementation of Graph Convolutional Collaborative Filtering proposed in our paper:<br>
*Minkyu Kim and Jinho Kim, Collaborative Filtering on Bipartite Graphs using Graph Convolutional Networks (BigComp 2022)*.


## Requirements
* PyTorch
* scikit-learn
* scipy
* numpy
* pandas
* absl
* h5py

## Usage
```
python main.py --data_name=<dataset name> --root_dir=<dataset directory path>
```

## Citation
```
@inproceedings{
  kim2022collaborative,
  title={Collaborative Filtering on Bipartite Graphs using Graph Convolutional Networks},
  author={Kim, Minkyu and Kim, Jinho},
  booktitle={2022 IEEE International Conference on Big Data and Smart Computing (BigComp)},
  year={2022}
}
```