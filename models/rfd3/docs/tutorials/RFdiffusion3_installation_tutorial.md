# Installation of RFdiffusion3

## Table of Contents
- [Learning Objective](#learning-objective)
- [Prerequisites](#prerequisites)
- [Tutorial](#tutorial)
    - [Step 0 (optional): Requesting a GPU with SLURM](#step-0-optional-requesting-a-gpu-with-SLURM)
    - [Step 1: Creating a conda environment](#step-1-creating-a-conda-environment)
    - [Step 2: Installing RFdiffusion3](#step-2-installing-rfdiffusion3)
    - [Step 3: Verify the installation](#step-3-verify-the-installation)
- [GPU vs. CPU Execution](#gpu-vs-cpu-execution)
    - [How RFdiffusion3 detects and uses your GPU](#how-rfdiffusion3-detects-and-uses-your-gpu)
    - [Running on latest GPU architectures (e.g. Blackwell)](#running-on-latest-gpu-architectures-eg-blackwell)
- [Glossary](#glossary)
- [Resources & References](#resources--references)

## Learning Objective
By the end of this tutorial, you will be able to install RFdiffusion3 on a Unix-based system using <code>pip</code> and verify that the installation was successful by running a minimal test example. After completing this tutorial, you will have a working RFdiffusion3 environment capable of running basic design tasks and ready for use in downstream protein design workflows.

## Prerequisites  
RFdiffusion3 is supported on Unix-based systems (Linux or macOS). Windows is not officially supported unless used through a Linux subsystem (e.g., WSL2). While not strictly required, using an environment manager such as Conda (Anaconda, Miniconda or Miniforge) is strongly recommended to isolate dependencies and avoid conflicts with your system-wide Python installation. For practical protein design workloads, a machine equipped with an NVIDIA GPU with ~3 GB of memory is highly recommended, as GPU acceleration substantially reduces inference time. This requires a recent NVIDIA driver installation. RFdiffusion3 can also run on CPU-only systems, however, runtime may increase significantly depending on the design task.

List of requirements:
- Linux or macOS
- Python 3.9-3.12
- Enviroment manager such as Conda (Anaconda, Miniconda or Miniforge)
- A working internet connection (for downloading the model and weights)
- Sufficient disk space for the model [checkpoint](#checkpoint) (~3 GB)
- A downloading tool such as <code>wget</code> or <code>curl</code>
- Optional but recommended: An NVIDIA GPU with a recent driver installation and at least ~16GB of RAM


## Tutorial
### Step 0 (optional): Requesting a GPU with SLURM
If you want to run RFdiffusion3 on a GPU and are working on a shared scientific cluster, a GPU is usually not available by default. In that case, you must first request one through the cluster's job scheduler (e.g., [SLURM](#SLURM)). If you are installing RFdiffusion3 on your local machine, or if you plan to run it only on CPU, you can [skip](#step-1-creating-a-conda-environment) this step.

On many [SLURM](#SLURM) clusters, the exact command for allocating a GPU depends  on the local configuration. A general pattern is:
```
salloc --gres=gpu:1 --account=<account-name> --partition=<partition-name>
```
This requests:
- 1 GPU
- access under the specified account
- resources from the specified partition

After the allocation is granted, start an interactive shell with:
```
srun --pty $SHELL
```

If you cluster requires explicit CPU cores, system memory and a time limit, a more complete request may look like:
```
salloc --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=01:00:00 --account=<account-name> --partition=<partition-name>
```

Once the session has started, verify that a GPU was allocated by running:
```
nvidia-smi
```

If the allocation was successful, this command will display information such as the GPU model, the NVIDIA driver version and the current GPU memory usage. If you see <code>command not found</code>, or if no GPU is listed, then no GPU is available in your current session.

### Step 1: Creating a conda environment
Next, create an isolated [conda environment](#conda-environment). Anaconda or its lightweight variants, Miniconda and Miniforge, allow you to isolate Python and its associated libraries from your system-wide installation. This prevents dependency conflicts and ensures that your global Python environment remains unaffected. If any issues occur during installation, the environment can simply be removed without impacting the rest of your system.

In the terminal input the following command:  
```
conda create -n RFD3 python=3.12 -y
```

Here:
- <code>-n RFD3 </code>specifies the name of the environment.
- <code>python=3.12 </code>defines the Python version (a recent version like 3.9-3.12 is recommended).
- <code>-y </code>automatically confirms installation prompts

Once the creation is completed, activate the environment with:  
```
conda activate RFD3
```

You should now see <code>(RFD3)</code> prefixed in your terminal prompt, indicating that the environment is active. 
To verify that the correct Python interpreter is being used, run:  
```
which python
```
The displayed path should point to the newly created [conda environment](#conda-environment), for example:  
<code>/home/florian_wieser/miniconda3/envs/RFD3/bin/python</code>  

### Step 2: Installing RFdiffusion3
RFdiffusion3 is distributed as part of the <code>rc-foundry</code> Python package. [Foundry](#foundry) is the RosettaCommons framework that provides a unified [command-line interface](#cli) for running multiple protein modeling and design deep learning models. It includes RosettaFold3 for structure prediction, ProteinMPNN for inverse folding and RFdiffusion3 for generative protein design. While this tutorial focuses on RFdiffusion3, RosettaFold3 and ProteinMPNN can be installed in a similiar manner.  

Install RFdiffusion3 using:  
```
pip install "rc-foundry[rfd3]"
```
The quotation marks around <code>rc-foundry[rfd3]</code> are important in shells such as <code>zsh</code>, where square brackets have special meaning. Without quotes, the command may fail.

#### Dowloading the model checkpoint  
RFdiffusion3 requires a trained model file (a [checkpoint](#checkpoint)) containing the learned neural network weights (~3 GB).  
Download the [checkpoint](#checkpoint) file using:  
```
foundry install rfd3
```
By default, this command will download the [checkpoint](#checkpoint) to <code>~/.foundry/checkpoints</code>.  

Optional: If you prefer to store the [checkpoint](#checkpoint) in a custom location (for example, on a cluster with limited home directory space), you can specify a custom [checkpoint](#checkpoint) directory using the <code>--checkpoint-dir</code> flag:  
```
foundry install rfd3 --checkpoint-dir <path/to/checkpoint_dir>
```
This will download the [checkpoint](#checkpoint) to the specified directory and register that directory via the <code>FOUNDRY_CHECKPOINT_DIRS</code> [environment variable](#environment-variable), so that RFdiffusion3 automatically searches it in future runs (in addition to the default <code>~/.foundry/checkpoints</code> location).


### Step 3: Verify the installation
To verify that RFdiffusion3 was installed correctly, we will download and run a minimal demonstration example from the official RFdiffusion3 repository. First, create a working directory and a directory for the example input files, then download the required files:
```
mkdir -p input_pdbs  
mkdir -p verify_installation
wget -P verify_installation https://raw.githubusercontent.com/RosettaCommons/foundry/refs/heads/production/models/rfd3/docs/examples/demo.json
wget -P input_pdbs https://raw.githubusercontent.com/RosettaCommons/foundry/production/models/rfd3/docs/input_pdbs/M0255_1mg5.pdb  
wget -P input_pdbs https://raw.githubusercontent.com/RosettaCommons/foundry/production/models/rfd3/docs/input_pdbs/7v11.pdb  
wget -P input_pdbs https://raw.githubusercontent.com/RosettaCommons/foundry/production/models/rfd3/docs/input_pdbs/1bna.pdb
```
After downloading the files, enter the working directory and run the demo using:  
```
cd verify_installation
rfd3 out_dir=demo_output inputs=demo.json
```
The output directory (<code>demo_output</code>) will be created automatically if it does not exist already. On a modern GPU, this example typically completes within a few minutes. On a CPU, runtime may increase substantially depending on the hardware. Some warning messages during execution are normal and can be ignored.

Expected output:  
Inside <code>demo_output/</code>, you should find structure files (.cif.gz) for each demo example and summary score files (.json). If these were generated without errors, the installation was successful. You can expect the generated structures using a molecular visualization tool such as PyMOL, or examine the score files in a text editor.

# GPU vs CPU Execution
RFdiffusion3 can run on both CPUs and GPUs, but the performance difference is very large:
- CPU only: Suitable for small tests and learning. Inference will run, but it can be very slow — often tens of minutes per example or longer depending on your CPU.
- GPU: Strongly recommended for real use, especially with multiple targets. A modern NVIDIA GPU can reduce runtime from minutes to seconds per design.

## How RFdiffusion3 detects and uses your GPU
To run RFdiffusion3 on a GPU no additional setup steps are required. It will automatically try to use a GPU if:
1. A compatible NVIDIA GPU is present
2. A matching CUDA toolkit is installed
3. [pytorch](#PyTorch) can communicate with the GPU

You can verify your GPU availability inside your [conda environment](#conda-environment) using:
```
python - << 'EOF'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count(), "GPUs detected")
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
EOF
```
Make sure your Conda environment is activated before running this check.
If <code>CUDA available: True</code> is printed and a GPU name appears, [pytorch](#PyTorch) can access your GPU.

## Running on latest GPU architectures (e.g. Blackwell)
Very recent NVIDIA GPU architectures (such as Lovelace or Blackwell) may require:
- A sufficiently new NVIDIA driver
- A compatible CUDA runtime
- A recent [pytorch](#PyTorch) build that supports the architecture

In most cases, installing RFdiffusion3 as described above is sufficient. However, for very new GPUs, you may need to upgrade [pytorch](#PyTorch) to a more recent (or nightly) build that includes support for the latest CUDA versions.

Installation example:
```
# 1. Install rfd3 and its required dependencies
pip install "rc-foundry[rfd3]"

# 2. Upgrade to PyTorch nightly (cu128) to add Blackwell GPU support
pip install --pre torch torchvision torchaudio \
--index-url https://download.pytorch.org/whl/nightly/cu128 \
--upgrade --force-reinstall
```

Important notes:
- Replace cu128 with the CUDA version supported by your driver (check via <code>nvidia-smi</code>).
- Installing a nightly [pytorch](#PyTorch) build overrides the version originally installed with <code>rc-foundry</code>.
- After upgrading [pytorch](#PyTorch), verify GPU detection again using the [Python check](#how-rfdiffusion3-detects-and-uses-your-gpu) from above.

## Glossary
### <a id="checkpoint"></a>Checkpoint
A binary file that contains the trained model weights of a neural network model. For RFdiffusion3, this file stores the learned parameters the model needs for inference. Without a checkpoint, the model cannot run.

### <a id="foundry"></a>Foundry
A toolkit from RosettaCommons that provides a unified Python [CLI](#cli) and framework for running multiple machine learning-based protein modeling and design tools (e.g., RFdiffusion3, RosettaFold3, ProteinMPNN). It also manages model weights (“[checkpoints](#checkpoint)”) and common configurations.

### <a id="cli"></a>CLI (Command-Line Interface)
A way to interact with software by typing commands in a terminal. Foundry and RFdiffusion3 provide CLI commands like <code>foundry install rfd3</code> and <code>rfd3</code>

### <a id="conda-environment"></a>Conda Environment
An isolated Python environment created using Conda or Miniconda. It keeps project dependencies separate from your system Python to prevent version conflicts.

### <a id="environment-variable"></a>Environment Variable
A value stored in the shell session that software can read to determine paths or settings (e.g., <code>FOUNDRY_CHECKPOINT_DIRS</code>).

### <a id="SLURM"></a>SLURM
An open-source job scheduler used on high-performance computing (HPC) clusters to manage and distribute computational workloads. It allocates resources such as CPUs, GPUs, memory, and nodes, and runs user-submitted jobs through a queueing system.

### <a id="torch"></a>PyTorch
An open-source deep learning framework used to build and run neural network models. RFdiffusion3 is implemented using PyTorch. PyTorch handles tensor operations, GPU acceleration (via CUDA), and model inference.

## Resources & References
- RosettaCommons Foundry (GitHub Repository) https://github.com/RosettaCommons/foundry
- RFdiffusion Documentation https://github.com/RosettaCommons/foundry/tree/production/models/rfd3
- RFdiffusion3 publication https://www.science.org/doi/10.1126/science.ade2574
- PyMOL visualization software https://pymol.org/