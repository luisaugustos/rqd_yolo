# Training Guide for Remote Servers

This guide explains how to set up and run training on a remote server.

## Quick Start

### 1. Clone and Setup

```bash
# Clone repository
git clone <repository-url>
cd rqd_yolo

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file with Roboflow API key:

```bash
ROBOFLOW_API_KEY=your_api_key_here
```

### 3. Download Dataset

```bash
# Download from Roboflow
python scripts/data/download_roboflow.py

# Or use train script with --download flag
python train.py --download --sequential
```

### 4. Train Models

```bash
# Train single model
python train.py --model yolo12n --epochs 50

# Sequential training (yolo12n → yolo12m)
python train.py --sequential

# With custom batch size
python train.py --model yolo12m --batch 8 --epochs 50

# Specify GPU device
python train.py --sequential --device 0
```

## Advanced Usage

### Training Scripts

**Main Orchestration Script:**
- `train.py` - Entry point, supports downloading and sequential training

**Training Scripts in `scripts/training/`:**
- `train_roboflow_config.py` - Config-based training with portable paths
- `train_yolo12_sequential.py` - Sequential trainer for YOLO12 models
- `train_roboflow.py` - Main training script (in `scripts/`)

### Utility Scripts in `scripts/utils/`

```bash
# Compare model performance
python scripts/utils/compare_models.py

# Monitor training progress
python scripts/utils/monitor_training.py
```

### Data Scripts in `scripts/data/`

```bash
# Download dataset from Roboflow
python scripts/data/download_roboflow.py

# Auto-annotate images with trained model
python scripts/data/auto_annotate_dataset.py

# Upload images to Roboflow
python scripts/data/upload_roboflow_images.py
```

## Configuration

Edit `config.yaml` to customize:

- **Paths**: Data, models, results directories
- **Roboflow**: Workspace and project settings
- **Training**: Default epochs, batch size, device
- **Dataset**: Class names, split ratios
- **GPU**: Device selection and mixed precision

Example config modification:

```yaml
training:
  default_model: yolo12m
  default_epochs: 100
  default_batch: 8

gpu:
  device: 0
  mixed_precision: true
```

## Remote Server Setup (Linux/SSH)

### 1. Transfer Repository

```bash
# From local machine
rsync -avz --exclude '.git' --exclude 'runs' --exclude 'data/raw' . user@server:/path/to/rqd_yolo
```

### 2. Install on Server

```bash
# SSH into server
ssh user@server

# Navigate to project
cd /path/to/rqd_yolo

# Create environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure .env
echo "ROBOFLOW_API_KEY=your_key" > .env
```

### 3. Run Training

```bash
# Download dataset (one-time)
python train.py --download --sequential

# Or train directly if dataset already present
python train.py --sequential

# Detach and let run in background
nohup python train.py --sequential > training.log 2>&1 &
```

### 4. Monitor Progress

```bash
# Check training log
tail -f training.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# View results
python scripts/utils/monitor_training.py
```

## Typical Training Timeline

For YOLO12 models on RTX 3070 Ti (8GB):

| Model | Batch | Epochs | Time |
|-------|-------|--------|------|
| yolo12n | 16 | 50 | ~2 hours |
| yolo12m | 8 | 50 | ~3 hours |
| yolo26n | 16 | 50 | ~2.5 hours |

**Sequential (yolo12n + yolo12m): ~5-6 hours total**

## Troubleshooting

### Out of Memory (OOM)

If training fails with OOM:

1. Reduce batch size: `--batch 8`
2. Use smaller model: `--model yolo12n`
3. Reduce image size in config.yaml: `imgsz: 416`

### Dataset Not Found

```bash
# Check data paths
ls -la data/

# Download manually
python scripts/data/download_roboflow.py

# Or update paths in config.yaml
```

### GPU Not Detected

```bash
# Verify CUDA setup
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Results and Models

After training:

```
runs/detect/results/
├── yolo12n_rock-quality1/
│   ├── weights/best.pt          # Best model
│   ├── weights/last.pt          # Final checkpoint
│   └── results.csv              # Training metrics
├── yolo12m_rock-quality2/
└── yolo26n_rock-quality3/
```

### Deploy Trained Model

```bash
python rqd_cli.py --model runs/detect/results/yolo12n_rock-quality1/weights/best.pt --input image.jpg
```

## Best Practices

1. **Always use config-based paths** - Makes scripts portable across systems
2. **Version control configs** - Track changes to hyperparameters
3. **Save training logs** - Redirect output: `nohup python train.py ... > training.log 2>&1`
4. **Monitor GPU memory** - Adjust batch size if needed
5. **Backup results** - Copy `runs/` and `results/` periodically
6. **Test before large run** - Train one epoch: `python train.py --model yolo12n --epochs 1`

## Support

For issues with:
- **Roboflow dataset**: Check API key and workspace settings
- **PyTorch/CUDA**: Verify CUDA 12.1 installation
- **Model availability**: Ensure Ultralytics hub access
- **GPU errors**: Check device IDs and VRAM availability

See main `README.md` for full project documentation.
