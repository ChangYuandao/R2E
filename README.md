# Installation

Due to file upload size limitations, please download version 1.6.1 from https://github.com/isaac-sim/IsaacGymEnvs and place the `assets` folder in our `IsaacGymEnvs` directory.

First, ensure that IsaacGym is installed in your environment (tested with Preview Release 4). If you have the package, follow the commands below to install it:

```bash
tar -xvf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
pip install -e .
python examples/joint_monkey.py # (test installation)
```

Next, set up the conda environment by running the following commands:
```bash
conda create -n r2e python=3.8
conda activate r2e
```

After setting up the environment, install the required dependencies:

```bash
cd isaacgymenvs; pip install -e .
cd ../rl_games; pip install -e .
```

Then, configure the QWEN model API key by exporting it in your terminal:
```bash
export DASHSCOPE_API_KEY="YOUR_API_KEY"
```

# Running R2E

Once the setup is complete, navigate to the R2E directory and run the desired task. For example, to run the Ant task, execute the following command in the terminal:
```bash
python env=ant max_iterations=500 test_max_iterations=500
```
The required training and evaluation iterations for different tasks are specified in the `different_task_iterations.txt` file.

