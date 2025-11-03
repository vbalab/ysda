<!-- markdownlint-disable MD001 MD010 MD024 MD025 MD049 -->

# Workflow

```bash
ssh balabaevvl@45.93.203.34
ssh shad-gpu

# scp /path/to/local/file username@remote_host:/path/to/remote/destination
scp main.cu shad-gpu:cuda/
scp notes.md shad-gpu:cuda/

# scp username@remote_host:/path/to/remote/file /path/to/local/destination
scp -r shad-gpu:cuda/* .
```

```bash
nvidia-smi                                                              # see GPUs
nvidia-smi -L
```

```bash
tmux
tmux new -s isaac
tmux attach -t isaac
```

[Isaac Sim Installation Guide](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_python.html#isaac-sim-app-install-pip)

```bash
python3.11 -m venv env_isaacsim
source env_isaacsim/bin/activate

pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install isaacsim[all,extscache]==5.0.0 --extra-index-url https://pypi.nvidia.com

pip show isaacsim
```

```bash
isaacsim
```

Download [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/download.html#isaac-sim-latest-release) locally on your laptop.

```bash
isaacsim isaacsim.exp.full.streaming --no-window
isaacsim isaacsim.exp.full.streaming --no-window --/app/livestream/publicEndpointAddress=45.93.203.34 --/app/livestream/port=49100
```

```bash
ps -ef | grep isaacsim
kill -KILL ...
```

---

```bash
isaacsim ... 2>&1 | tee isaacsim.log    # server
scp shad-gpu:isaacsim.log .             # laptop
```
