# LiGO Minimal
Minimal implementation of LiGO (Learning to Grow Pretrained Models for Efficient Transformer Training))

### Installation
Create virtual environment and install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements
cd core
```

### Running

- Train a small model from scratch
```bash
cd core/
python3 main.py --model=fc1 --initializer=scratch
```

- Training deeper and wider model, with zero-shot weight transfer initialization
```bash
python3 main.py --model=fc2 --initializer=ligo --pretrain_model=fc1_scratch
```
