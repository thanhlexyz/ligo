# LiGO Minimal
Minimal implementation of [LiGO (Learning to Grow Pretrained Models for Efficient Transformer Training)](https://arxiv.org/abs/2303.00980)

### Installation
Create virtual environment and install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements
cd core
```

### Usage

- Train a small FC model from scratch for MNIST
```bash
cd core/
python3 main.py --model=fc1 --initializer=scratch
```

- Training deeper and wider FC model, with LiGO initialization from small model for MNIST
```bash
python3 main.py --model=fc2 --initializer=ligo --pretrain_model=fc1_scratch
```

- The main implementation is in [`ligo.py`](./core/lib/initializer/ligo.py)
