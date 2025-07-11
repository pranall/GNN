# GNN Extension for Diversify on EMG Dataset

## 1. Project Overview

This project is a **Graph Neural Network (GNN)** extension of the [DIVERSIFY](https://github.com/microsoft/robustlearn/tree/main/diversify) framework, originally designed for domain generalization in time-series classification. The main objective was to apply a GNN-based approach to Electromyography (EMG) data in cross-subject settings and compare performance to the original CNN-based Diversify algorithm.

### What is a GNN?

A **Graph Neural Network** is a type of neural network that operates on graph structures instead of regular grids (like images) or sequences (like text/time-series).
**Key characteristics:**

* Nodes = entities (e.g., EMG channels/sensors)
* Edges = relationships (e.g., correlations between sensors)
* GNNs can capture interactions between nodes through message passing.
* Common GNN layer used here: **GCNConv** (Graph Convolutional Network layer).
* Our model includes both **temporal convolutions** and **graph convolutions**.

### The EMG Dataset and Motivation

* **EMG data**: Multi-channel 1D time-series signals from muscle activity, typically structured as `(num_samples, num_channels, time_steps)` (e.g., 8 channels × 200 timesteps).
* **Motivation**: GNNs are strong in domains with explicit relational structure. We hypothesized that modeling correlations between EMG channels as graphs could enhance cross-subject generalization, which is challenging for CNNs due to domain shifts.

---
### GNN with EMG Dataset Pipeline as presented in this project

<img width="1536" height="1024" alt="GNN Pipeline" src="https://github.com/user-attachments/assets/8e5bcf3f-6ad5-4e72-855d-ca7ace4e04b1" />

Dataset can be downloaded from this link: (https://wjdcloud.blob.core.windows.net/dataset/diversity_emg.zip)

## 2. Repository Structure & Usage



### File/Folders Explained
This GitHub repo consists of 8 directories and 31 files which are presented below as a tree diagram with each file's purpose.

```bash
.
├── alg/                       # Training algorithm logic
│   ├── alg.py                 # Main algorithm manager for training/evaluation
│   ├── algs/
│   │   ├── base.py            # Abstract base class for algorithms
│   │   ├── diversify.py       # Diversify algorithm implementation (baseline)
│   │   └── __init__.py
│   ├── __init__.py
│   ├── modelopera.py          # Model loading and saving utilities
│   └── opt.py                 # Custom optimizers
├── datautil/
│   ├── actdata/
│   │   ├── cross_people.py    # Data loader for cross-subject EMG experiments
│   │   ├── __init__.py
│   │   └── util.py            # Activity data transforms
│   ├── getdataloader_single.py# Builds DataLoaders for training/validation/testing (GNN & CNN)
│   ├── graph_utils.py         # Graph construction/conversion tools for GNN
│   ├── __init__.py
│   └── util.py                # General data utilities
├── emg_gnn.yml                # Conda environment setup for GNN experiments
├── env.yml                    # (Optional) Another environment setup
├── gnn/
│   ├── graph_builder.py       # Build graph data from EMG signals
│   ├── __init__.py
│   └── temporal_gcn.py        # Temporal GCN (spatio-temporal model for EMG)
├── loss/
│   ├── common_loss.py         # Loss functions for training
│   └── __init__.py
├── network/
│   ├── act_network.py         # Activity recognition network modules
│   ├── Adver_network.py       # Domain adversarial network modules
│   ├── common_network.py      # Shared/common network layers
│   └── __init__.py
├── README.md                  # Project documentation (this file)
├── requirements.txt           # Python package dependencies
├── train.py                   # Main entry point for training/evaluation
└── utils/
    ├── __init__.py
    ├── params.py              # Experiment parameters and argument parsing
    └── util.py                # Miscellaneous helper utilities
```


### How to Run

Google colab was used to implement this project.
**Environment setup:**

install requirements:

```bash
pip install -r requirements.txt
```

Or in the .ipynb file paste the following code in the first code block
```bash
# First, uninstall existing PyTorch to avoid conflicts
!pip uninstall torch torchvision torchaudio -y

# Install PyTorch 2.1.0 with CUDA 11.8 (latest stable version)
!pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric (PyG) dependencies (FOR TORCH 2.1.0 + CUDA 11.8)
!pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
!pip install torch-geometric

# Install other required packages
!pip install fastdtw numpy==1.24.3
```

Restarting the file is a must after implemeting these import statements. These imports help resolve most of the environment issues for a smoother run.

**Train the GNN model:**
Four commands are provided as follows:

```bash
!python train.py --data_dir ./data/ --task cross_people --dataset emg --max_epoch 1 --local_epoch 3 --lr 0.001 --batch_size 32 --seed 42 --output ./data/train_output/ --use_gnn --model_type gnn --test_envs 0
```
```bash
!python train.py --data_dir ./data/ --task cross_people --dataset emg --max_epoch 5 --local_epoch 3 --lr 0.001 --batch_size 32 --seed 42 --output ./data/train_output/ --use_gnn --model_type gnn --test_envs 1
```
```bash
!python train.py --data_dir ./data/ --task cross_people --dataset emg --max_epoch 10 --local_epoch 3 --lr 0.001 --batch_size 32 --seed 42 --output ./data/train_output/ --use_gnn --model_type gnn --test_envs 2
```
```bash
!python train.py --data_dir ./data/ --task cross_people --dataset emg --max_epoch 15 --local_epoch 3 --lr 0.001 --batch_size 32 --seed 42 --output ./data/train_output/ --use_gnn --model_type gnn --test_envs 3
```
### Results

* **CNN (Diversify)** achieves **80–85%** accuracy on cross-subject EMG recognition.
* **GNN extension** achieved only **16–18%** accuracy (depending on setup) with unstable training and high loss.
  *(Your results may vary, but the core takeaway is that GNNs struggled to generalize.)*

---

## 3. Why GNN Results Were Low
Electromyography (EMG) is a technique for recording the electrical activity of muscles using multiple electrodes (channels) placed on the body. Its purpose is to analyze muscle activity for gesture recognition, prosthetic control, rehabilitation, or medical diagnostics. On the other hand, Graph Neural Networks (GNNs) are designed for data that is inherently graph structured. In GNNs nodes represent entities (e.g., people, atoms), edges represent relationships (e.g., friendships, chemical bonds) and GNNs excel when these relationships are irregular or complex. GNNs seem to be incompatible with a dataset like EMG since there is no relation between the channels in EMG (muscles have no relation with other muscles and no bonds for GNN to function well). Hence, the results of this extension were always low to moderate.

This pictorial representation depeicts how GNN behaves with EMG dataset. It is a representation of EMG data, Natural Graph and EMG as a forced graph. The figure depicts that EMG is a linear time series data which was forced to have connections (synthetic) for GNN to run.

<img width="1536" height="1024" alt="GNN EMG" src="https://github.com/user-attachments/assets/bef6004f-02e1-4f02-9655-c2630d8dfcce" />

The following tabular information is based on the code of this repo and its analysis. 

Table 1: Core Implementation Decisions

| **Component**      | **Approach**                                                                      |
| ------------------ | --------------------------------------------------------------------------------- |
| Data Structure     | Samples: \[8 channels, 1, 200 timesteps] reshaped to \[8, 200], then graphified   |
| Graph Builder      | Correlation-based edge construction (adaptive thresholding, self-loops if needed) |
| Precomputation     | Graphs precomputed and cached as `.pt` files for speed                            |
| Train/Val/Test     | Careful splits: Cross-person (train/val/test splits by person)                    |
| GNN Model          | `TemporalGCN`: Spatial (GCNConv) + Temporal (Conv1d/MaxPool) layers               |
| Metrics            | Accuracy (main), training/validation/test splits, and class distributions printed |
| Evaluation         | Tracked epoch-wise metrics, time, and best validation accuracy                    |


Table 2: Training Regimen & Compute Analysis. 

| **Run Configuration**    | **Parameters**                                                         |
| ------------------------ | ---------------------------------------------------------------------- |
| Batch Size               | 32                                                                     |
| Epochs**                 | 1, 5, 10, 15 (For test environemnts 0, 1, 2 and 3)                     |
| Device                   | GPU (Colab: T4, also analyzed: JetStream2 VM with 8 CPUs, 30GB RAM)    |
| Precomputation           | \~15 minutes                                                           |
| Training (5 epochs)      | \~1.5 hours                                                            |
| Training (15 epochs)     | \~4.5 hours (estimate)                                                 |

It must be noted that since the accuracy is stagnant, running on more epochs will only waste the computational resources. Hence the epochs numbers are set low. Also since the computation required is very heavy, Colab either signs you out or disconnects due to Colab usage limits which is a similar case with JetStream2

Table 3: Limitation Analysis
| **Aspect**             | **Observation / Challenge**                                                                     | **Mitigation / Notes**                                |
| ---------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| **Accuracy**           | GNN achieved 16-17% accuracy (vs. 80%+ for CNN baseline).                                       | GNNs not optimal for this EMG dataset’s structure.    |
| **Data Shape**         | EMG better fits regular grid (suitable for CNNs), not an inherent graph structure.              | Justified as a research experiment.                   |
| **Resource Limits**    | Colab usage limits interrupted long training sessions.                                          | VM/cloud solution considered.                         |

Table 4: Justifications for Design Choices

| **Decision**                  | **Justification**                                                                        |
| ----------------------------- | ---------------------------------------------------------------------------------------- |
| Trying GNNs for EMG           | Research motivation: Investigate if GNNs can capture latent correlations in EMG channels |
| Precompute and cache graphs   | Practical necessity for large datasets and repeated runs                                 |
| Printing all metrics and logs | Transparency for debugging, tracking splits, and ensuring integrity of evaluation        |
| Extensive documentation       | To clearly show every step, decision, and challenge for future readers/reviewers         |
| Including pipeline diagram    | For clarity in presentations and reports; makes workflow and data flow easily digestible |


**Conclusion: GNNs simply aren’t a good fit for EMG time-series data. Here’s why:**

* **No natural graph structure:** EMG channels are sequentially sampled 1D signals—not entities with strong, fixed, pairwise relationships. Graph construction was forced, often using arbitrary or unstable correlation thresholds. Muscles have no relation with each other neither do they have a bond which is a basic requirement for GNN to run.
* **Architectural mismatch:** GNNs shine when the problem is about *relationships* between nodes (e.g., social networks, molecules). For EMG, CNNs or TCNs (Temporal Convolutional Networks) better capture the temporal and spatial patterns.
* **Resource hungry:** Precomputing graphs and running GNNs on GPU consumed excessive Colab/GPU resources with little payoff.
* Despite their theoretical appeal for modeling dependencies in multichannel sensor data, practical results show GNNs are not competitive compared to CNNs for this EMG dataset, which has a regular, grid-like structure better suited for convolutional architectures.
* GNNs could potentially be effective only if the graph construction accurately captures true muscle interactions beyond simple linear correlations and node and edge features incorporate physiologically meaningful information
* Since the EMG dataset’s correlations are largely linear, GNN accuracies remain flat and linear as well.

**Bottom line:**
*CNNs remain the practical and reliable choice for cross-subject EMG, offering higher accuracy and stability. GNNs, while theoretically interesting, introduce unnecessary complexity without meaningful gains for this type of signal. However, improving the code pipeline will not overcome the fundamental mismatch—a bicycle cannot be upgraded to swim in a pool. Forced graph construction here does not reflect true muscle synergies and thus limits GNN applicability.*

---
### Technical Section.
This section deeply explains the technical challenges faced while coding. This also serves as a guide of 'if you ever want to try to run GNN on EMG dataset, do not make these mistakes.' 
Top issues faced while preparing this code and Repo:
1) Import errors
2) Environmental issues (mismatch of environments as Pytorch is used and ut requires the latest updated version to support graphs)
3) Shape mismatchs [EMG datset is [8,1,200] but for graph we had to sueeze out 1 and reverse it to [200,8] for smoother graph construction.
4) Pre computing graphs
5) Trying to adjust Cache so as to not waste the resources.
   
However, this Github Repo is well equipped with good and robust pipeline which has eliminated stabilized all the above issues which enable the script to run very smoothly.

Below are the pictures which depict the outcomes in the logs for more clear understanding of the pipeline and how GNNs actually worked on EMG datset. Most of them were print statements which are still present in the code, but right now are commented out so as to keep the print log clean. The output explanations are done in parts.

1) Output 1
   
![GNNOutput1](https://github.com/user-attachments/assets/484a7a9c-2db2-466f-ae0f-10cae2702aca)

Explanation: This output is a debug log snapshot showing your current environment setup and a key configuration parameter:

* **\[DEBUG] Using graph\_threshold: -1.0**
  Indicates that the graph construction threshold is set to -1.0, meaning no threshold filtering is applied on graph edges.

* **Environment:**
  Lists versions of important software components used in your ML project:

  * Python 3.11.13
  * PyTorch 2.1.0 with CUDA 11.8 support
  * Torchvision 0.16.0 compatible with PyTorch
  * CUDA version 11.8
  * cuDNN 8700 (NVIDIA’s deep learning library)
  * NumPy 1.24.3 (numerical computing)
  * PIL 11.2.1 (image processing)

* The separator line `===========================================` marks the end of this info block.

**In brief:** This confirms your training environment details and that your graph threshold is disabled, which is essential for reproducibility and debugging.

2) Output 2
   
![GNNOutput2](https://github.com/user-attachments/assets/d20210dc-4563-46dd-8d0e-f3002d164e13)

Explanation: This output shows the key **training configuration and hyperparameters** for your GNN model run. Here’s a brief explanation:

* **model\_type: gnn** — Using a Graph Neural Network architecture.
* **algorithm: diversify** — The domain generalization algorithm applied.
* **alpha, alpha1: 0.1** — Loss weights or regularization parameters controlling contributions of different terms.
* **batch\_size: 32** — Number of samples processed simultaneously.
* **beta1: 0.5** — Likely optimizer’s momentum or Adam beta parameter.
* **checkpoint\_freq: 100** — Save model checkpoint every 100 iterations/steps.
* **local\_epoch: 3, max\_epoch: 5** — Number of inner loop training epochs and total epochs respectively.
* **lr: 0.001** — Learning rate for optimizer.
* **lr\_decay1, lr\_decay2: 1.0** — Learning rate decay multipliers (no decay applied here).
* **weight\_decay: 0.0005** — Regularization factor to prevent overfitting.
* **dropout: 0.0** — No dropout used.
* **label\_smoothing: 0.0** — No label smoothing applied.
* **bottleneck: 256** — Dimensionality of bottleneck layer.
* **classifier: linear** — Final classifier type is a linear layer.
* **dis\_hidden: 256** — Hidden units in domain discriminator.
* **layer: bn** — Batch normalization is used.
* **model\_size: median** — Model complexity size preset.
* **lam: 0.0** — Lambda, possibly adversarial loss weight, set to zero here.
* **latent\_domain\_num: 4** — Number of latent domains modeled for domain generalization.
* **domain\_num: 0** — No explicit domain ID specified here.
* **dataset: emg** — Dataset used is EMG signals.

**In short:** This block lists all essential training hyperparameters controlling model architecture, optimization, domain adaptation, and data used in your GNN experiment.

3) Output 3
   
![GNNOutput3](https://github.com/user-attachments/assets/23997804-7898-41f9-9437-0db0f6af6afd)

Explanation: This output displays additional configuration settings related to data loading, GNN-specific parameters, and environment setup for your training run:

* **data\_dir: ./data/** — Directory path where the dataset is stored.
* **task: cross\_people** — Specifies the task is cross-subject learning.
* **test\_envs: \[0]** — Environment/domain ID 0 is held out as test set.
* **N\_WORKERS: 2** — Number of parallel data loader workers.
* **automated\_k: False** — Auto-selection of some parameter disabled.
* **curriculum: False** — Curriculum learning is turned off.
* **CL\_PHASE\_EPOCHS: 5** — Curriculum learning phase length (not used here).
* **enable\_shap: False** — SHAP explainability not enabled.
* **resume: None** — Training does not resume from checkpoint.
* **debug\_mode: False** — Debugging mode off.
* **use\_gnn: True** — GNN model is enabled.
* **gnn\_hidden\_dim: 32** — GNN hidden layer dimension.
* **gnn\_output\_dim: 128** — GNN output embedding dimension.
* **gnn\_lr: 0.001** — Learning rate for GNN optimizer.
* **gnn\_weight\_decay: 0.0001** — Weight decay (L2 regularization) for GNN.
* **gnn\_pretrain\_epochs: 5** — Number of pretraining epochs for GNN.
* **gpu\_id: 0** — GPU device ID used for training.
* **seed: 42** — Random seed for reproducibility.
* **output: ./data/train\_output/** — Directory to save training outputs.
* **old: False** — Indicates whether this is an old experiment/config.
* **steps\_per\_epoch: 100** — Number of training steps per epoch.
* **select\_position: {'emg': \[0]}** — Selected position/sensor index in EMG data.
* **select\_channel: {'emg': array(\[0,1,2,3,4,5,6,7])}** — Channels selected from EMG signals.
* **hz\_list: {'emg': 1000}** — Sampling frequency (Hz) for EMG data.

**In brief:** This block defines data paths, task setup, GNN model dimensions, optimization details, random seed, hardware configuration, and EMG data selection parameters controlling your experiment environment and training process.

4) Output 4
   
![GNNOutput4](https://github.com/user-attachments/assets/83ab492f-dc0d-47ce-a866-793bd9a85df9)

Explanation: This snippet provides final data and model input details:

* **select\_position & select\_channel:** Specifies which sensor positions and EMG channels (0 to 7) are used.
* **hz\_list:** Sampling frequency of 1000 Hz for EMG signals.
* **act\_people:** Groups of participant IDs used for activity data splitting.
* **input\_shape:** Input tensor shape is (8 channels, 1, 200 timesteps).
* **num\_classes:** 6 different classes in the classification task.
* **grid\_size:** Set to 10 (likely related to spatial processing or visualization).
* **graph\_threshold:** -1.0 means no threshold filtering on graph edges during graph construction.

In brief: This config details the EMG data channels, participants, input dimensions, number of classes, and graph construction parameters used in the experiment.


5) Output 5
   
![GNNOutput5](https://github.com/user-attachments/assets/158ba1b6-2821-4b69-b3db-59a61d3961dc)

Explanation: This output shows that:

* The training is running on a **CUDA-enabled GPU**.
* The **TemporalGCN (a GNN model)** is successfully activated for training.
* Raw data shapes are:

  * `x`: 6883 samples, 8 channels, 200 timesteps each
  * `cy`, `py`, `sy`: label arrays (likely class, person, session), each with 6883 entries.

In brief: Model training on GPU is ready, with input EMG data and corresponding labels properly loaded.

6) Output 6
   
![GNNOutput6](https://github.com/user-attachments/assets/de7e7557-8776-4f10-8418-c5ff1dc2e951)

Explanation: This output shows the incremental aggregation of EMG samples from multiple participants during dataset loading:

* For each person (e.g., Person 0, Person 1, …), it logs the number of initial samples.
* After adding each person’s data, it updates the cumulative dataset shape (`self.x.shape`) and label count (`self.labels.shape`).
* The shape format is `(number_of_samples, 8 channels, 200 timesteps)`.

**In brief:** The dataset is being built by sequentially appending samples from each person, tracking the growing size of input data and labels.


7) Output 7
   
![GNNOutput7](https://github.com/user-attachments/assets/f9bb7021-1d45-4764-b640-911590b8743c)

Explanation: This output continues showing dataset aggregation across participants:

* Each line logs the initial number of samples per person (e.g., Person 12 has 292 samples).
* After adding their data, it updates the total dataset shape and label count accordingly (e.g., cumulative 2581 samples after Person 12).
* Data shape consistently remains `(number_of_samples, 8 channels, 200 timesteps)`.

**In brief:** The dataset is progressively built by appending samples from each participant, keeping track of the growing total number of samples and labels.


8) Output 8
   
![GNNOutput8](https://github.com/user-attachments/assets/1f73093e-6cad-4936-b35f-1966d973e666)

Explanation: This output completes the dataset aggregation process by adding samples from participants 24 through 35:

* Logs each person’s initial sample count (e.g., Person 24 has 164 samples).
* Updates cumulative dataset size and label count after each addition, ending at 6883 total samples.
* The data shape remains consistent: `(samples, 8 channels, 200 timesteps)`.

**In brief:** Final portion of dataset loading showing incremental build-up of the full EMG dataset with sample counts tracked per person.


9) Output 9
    
![GNNOutput9](https://github.com/user-attachments/assets/df7cf8df-dd35-4265-83db-1ad1dfd57c94)

Explanation: This output shows:

* The dataset is fully loaded with **6883 samples**.
* **Memory usage is low (0 MB allocated/reserved)** at this point.
* A **graph cache miss** means no precomputed graphs found, so graphs need to be generated.
* **Precomputing graphs** starts: 27 batches processed, each batch reshapes input tensor from (8,1,200) to (200,8) for graph construction.
* All batches processed in \~15 minutes total.
* **6883 precomputed graphs saved** for future faster loading.
* Memory remains low after graph computation.

**In brief:** Graphs required for GNN training were not cached, so they were computed from raw data, saved, and memory stayed minimal throughout.


10) Output 10

![GNNOutput10](https://github.com/user-attachments/assets/1e113564-8c4c-47d0-ba90-b4e12c0674db)

Explanation: This output shows:

* Memory usage remains minimal (0 MB allocated/reserved).
* A **graph cache HIT** indicates precomputed graphs were found and loaded successfully from the file `./data/train_output/precomputed_graphs.pt`.
* All **6883 precomputed graphs** needed for training have been loaded.

**In brief:** Instead of recomputing, the program efficiently loads previously saved graph data, speeding up the training preparation.


11) Output 11

![GNNOutput11](https://github.com/user-attachments/assets/6a4edcc4-d145-4505-8f1d-38d5036ab475)

Explanation: This output provides dataset class distribution and split information:

* **Class distribution:** Shows counts of samples per class for training and validation/test sets.

  * For example, training classes have roughly 3300–3500 samples each; validation classes have about 1100–1175 samples each.
* **Train samples:** 4143 samples used for training.
* **Validation samples:** 1036 samples used for validation.
* **Target samples:** 1704 samples reserved for final testing or target domain evaluation.

**In brief:** Dataset is split into train, validation, and target sets with roughly balanced classes across all splits.


12) Output 12
    
![GNNOutput12](https://github.com/user-attachments/assets/6598ff11-a62c-4ec8-8c4a-78014abc9cc9)

Explanation: This output shows that:

* The GNN model is enabled and preparing for training.
* The program did not find precomputed graphs in cache (**Graph cache MISS**), so it’s generating them now.
* It is currently processing batch 18 out of 27 (67% done).
* Each batch takes about 33 seconds to process.

**In brief:** GNN training setup includes computing graph data on-the-fly due to missing cache, and it’s over halfway through this step.


13) Output 13
    
![GNNOutput13](https://github.com/user-attachments/assets/f755d4ab-f66b-4703-9baf-81a5f3e2e905)

Explanation: This message means:

* You **cannot access a GPU** on Google Colab right now because you’ve reached your GPU usage limits.
* To get more GPU access, you need to **purchase additional compute units** via “Pay As You Go.”
* Alternatively, you can continue running your notebook **without GPU**, but it will be much slower.
* Mostly happens cause the computational resources required to convert EMG dataset into graphs are too high, hence less epochs are preffered.

**In brief:** GPU resources are temporarily unavailable on Colab due to usage limits.


14) Output 14
    
![GNNOutput14](https://github.com/user-attachments/assets/d5edb763-c233-4090-8e1d-367eecd4f7a0)

Explanation: This screenshot shows that Google Colab cannot connect to a GPU because your usage limits have been reached.

* You are temporarily blocked from accessing GPU resources due to high usage.
* To get more GPU access, you can purchase Colab compute units via "Pay As You Go."
* Alternatively, you can continue running your notebook without GPU, but it will run slower.
* It must be noted that this is a virual JetStream2 which eventually timed out after completing 16 epcohs out of 20 epochs.

**In brief:** Your Colab session is out of GPU quota; no GPU is available until limits reset or you pay for more usage.

15) Output 15
    
![GNNOutput16](https://github.com/user-attachments/assets/e5384984-5e3b-44d8-a1f9-c3b0c152b6bc)

Explanation: A snippet of traning log whoch completed all 5 epochs.

16) Output 16
![GNNOutput15](https://github.com/user-attachments/assets/7529a345-1b58-4875-a7ce-be94b3af400e)

Explanation: A snippet of traning log whoch completed all 15 epochs.

It must be noted that since one epoch took around 18 minutes to complete, higher epochs like 10,15 and 20 epochs took enormous time to compute which is why the same .ipynb file was run simultaneously in three different colab with one including the virtual JetStream2 Colab whoch posed as a drawback since all the logs are not in one single .ipynb file. 
Usage of less epochs is highly recommended. 

This README file offers a full-proof guide on how a GNN is ran on EMG dataset. It must be noted that in this case, neither GNN is incorrect and neither is EMG dataset (it is well processed and is just fine). The key issue lies their individual architecture. In the most simple terms, GNN and EMG are like oil and water; no matter how much we mix them, they can never be mixed due to their properties. 


## Acknowledgement and References 

```bash
Inspired by [Diversify: Domain Generalization via Diversity Regularization](https://github.com/microsoft/robustlearn/tree/main/diversify)

@inproceedings{lu2022out,
  title={Out-of-distribution Representation Learning for Time Series Classification},
  author={Lu, Wang and Wang, Jindong and Sun, Xinwei and Chen, Yiqiang and Xie, Xing},
  booktitle={International Conference on Learning Representations},
  year={2023}
}

Thanks to Rishabh Gupta for contributing some code for this project. Please check his GitHub repo: (https://github.com/rishabharizona/gnnintegrated.git)
```
## Contact

```
If you find this information useful or would like to propose any suggestions feel free to drop an email here: (pranal.a.gaikwad@gmail.com)
```
