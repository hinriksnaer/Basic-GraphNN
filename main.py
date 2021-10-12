import os

import torch

import torch_geometric
import torch_geometric.data as geom_data

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


from trainer import NodeLevelGNN, GraphLevelGNN

CHECKPOINT_PATH = './checkpoint'
DATASET_PATH = './data'

device = 'cuda' if torch.cuda.is_available else 'cpu'

def train_node_classifier(model_name, dataset, **model_kwargs):
    pl.seed_everything(42)
    node_data_loader = geom_data.DataLoader(dataset, batch_size=1)#, num_workers=1)

    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "NodeLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=500,
                         progress_bar_refresh_rate=1) # 0 because epoch size is 1
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"NodeLevel{model_name}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = NodeLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything()
        model = NodeLevelGNN(model_name=model_name, c_in=dataset.num_node_features, c_out=dataset.num_classes, **model_kwargs)
        trainer.fit(model, node_data_loader, node_data_loader)
        model = NodeLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on the test set
    test_result = trainer.test(model, test_dataloaders=node_data_loader, verbose=False)
    batch = next(iter(node_data_loader))
    batch = batch.to(model.device)
    _, train_acc = model.forward(batch, mode="train")
    _, val_acc = model.forward(batch, mode="val")
    result = {"train": train_acc,
              "val": val_acc,
              "test": test_result[0]['test_acc']}
    return model, result

def train_graph_classifier(model_name, dataset, **model_kwargs):
    pl.seed_everything(42)

    dataset.shuffle()
    train_dataset = dataset[:150]
    test_dataset = dataset[150:]

    graph_train_loader = geom_data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    graph_val_loader = geom_data.DataLoader(test_dataset, batch_size=64) # Additional loader if you want to change to a larger dataset
    graph_test_loader = geom_data.DataLoader(test_dataset, batch_size=64)

    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "GraphLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=500,
                         progress_bar_refresh_rate=1)
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"GraphLevel{model_name}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = GraphLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)
        model = GraphLevelGNN(c_in=dataset.num_node_features,
                              c_out=1 if dataset.num_classes==2 else dataset.num_classes,
                              **model_kwargs)
        trainer.fit(model, graph_train_loader, graph_val_loader)
        model = GraphLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    # Test best model on validation and test set
    train_result = trainer.test(model, test_dataloaders=graph_train_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=graph_test_loader, verbose=False)
    result = {"test": test_result[0]['test_acc'], "train": train_result[0]['test_acc']}
    return model, result

# Small function for printing the test scores
def print_results(result_dict):
    if "train" in result_dict:
        print(f"Train accuracy: {(100.0*result_dict['train']):4.2f}%")
    if "val" in result_dict:
        print(f"Val accuracy:   {(100.0*result_dict['val']):4.2f}%")
    print(f"Test accuracy:  {(100.0*result_dict['test']):4.2f}%")

def train_mlp():

    cora_dataset = torch_geometric.datasets.Planetoid(root=DATASET_PATH, name="Cora")
    node_mlp_model, node_mlp_result = train_node_classifier(model_name="MLP",
                                                            dataset=cora_dataset,
                                                            c_hidden=16,
                                                            num_layers=2,
                                                            dp_rate=0.1)
    
    print_results(node_mlp_result)

def train_graphNN():

    cora_dataset = torch_geometric.datasets.Planetoid(root=DATASET_PATH, name="Cora")
    node_gnn_model, node_gnn_result = train_node_classifier(model_name="GNN",
                                                        layer_name="GCN",
                                                        dataset=cora_dataset,
                                                        c_hidden=16,
                                                        num_layers=2,
                                                        dp_rate=0.1)
    print_results(node_gnn_result)

def train_graph_level():
    tu_dataset = torch_geometric.datasets.TUDataset(root=DATASET_PATH, name="MUTAG")


    model, result = train_graph_classifier(model_name="GraphConv",
                                        dataset=tu_dataset,
                                        c_hidden=256,
                                        layer_name="GraphConv",
                                        num_layers=3,
                                        dp_rate_linear=0.5,
                                        dp_rate=0.0)

    print(f"Train performance: {100.0*result['train']:4.2f}%")
    print(f"Test performance:  {100.0*result['test']:4.2f}%")

if __name__ == '__main__':
    train_graph_level()