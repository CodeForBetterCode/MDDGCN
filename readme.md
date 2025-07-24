# MDDGCN

**MDDGCN** is a graph convolutional network (GCN)-based framework for identifying depression-related pathogenic genes.

# Requirements

```
python==3.8
torch==1.12.1
torch-geometric==2.6.1
scikit-learn==1.3.2
```

⚠️ Note: The `torch` and `torch-geometric` versions should be compatible with your CUDA version. Adjust accordingly if needed.

# Usage

## Data

The dataset can be constructed using the `utils.py` script.

## Simple usage

After installing the required packages, you can quickly run the model and test the model with the following command:

```
python MDDGCN.py
```

You can also adjust model settings via command-line arguments. Commonly used parameters include:

```
--modeltype              # Model to use (e.g., GCN, GAT, MDDGCN, etc.)
--epochs                 # Number of training epochs
--learning_rate          # Learning rate
--weight_decay           # Weight decay
--hidden_channels        # Hidden layer size
--feature                # Feature type (Biological / Expression / Enhanced)
--scheduler              # Learning rate scheduler
--perturb_features_p     # Feature perturbation probability
--dropout_edge_p         # Edge dropout probability
```

## Using Your Own Data

To use your own dataset, make sure it follows the PyTorch Geometric `Data` object format:

`data.x`: Node features (tensor of shape `[num_nodes, num_features]`)

`data.edge_index`: Graph connectivity (tensor of shape `[2, num_edges]`)

`data.y`: Node labels (tensor of shape `[num_nodes]`)

You can construct the graph using known PPI networks, such as those available from the [STRING database](https://string-db.org/), and pair them with your own node features and labels.

##  Train your own model

Train the model on your own dataset with:

```
python MDDGCN.py
```

Feel free to modify `MDDGCN.py` and `utils.py` to adapt the framework to your specific task or dataset.

#### 🔍 Predicting Unlabeled Genes

To train on known genes and predict unknown ones:

```
python MDDGCN_Train_Predict.py
```

# Other details

The other details can be seen in the paper and the codes.
