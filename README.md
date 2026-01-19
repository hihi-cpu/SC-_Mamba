🧠 SC²-Mamba

# 🚦SC²-Mamba: Semantic-Causal Clustering Mamba for Cross-City Traffic Flow Prediction

> A traffic flow prediction project based on the **Mamba architecture**, supporting **multi-granularity spatial-causal prediction** and **cross-city forecasting**, to achieve accurate and efficient traffic prediction.

---

## 📁 Project Structure

```text
MIGC-CMmamba/
├── Mamba/                      # Mamba-related modules
│   ├── data/                  # Data processing
│   │   ├── processed/        # Preprocessed data
│   │   ├── CHI/              # Chicago dataset
│   │   ├── DC/               # Washington D.C. dataset
│   │   └── NY/               # New York dataset
│   ├── file/                 # Core files
│   │   ├── Domain.py         # Domain adaptation module
│   │   └── Pretrain.py       # Pre-training module
│   └── models/               # Model files
│       ├── Mam.py            # Main Mamba model
│       ├── model_utils.py    # Model utility functions
│       ├── Softclustering.py # Soft clustering module
│       └── targetfinetune.py # Target domain fine-tuning
├── mainCHIDCbike.py          # Main script (Chicago→D.C. bike prediction)
├── mainNYDCtaxi.py          # Main script (New York→D.C. taxi prediction)
```

## ⚙️ Environment Configuration

| Dependency | Version |
| --- | --- |
| Python | 3.8.20 |
| PyTorch | 2.2.2 |
| mamba-ssm | 1.1.3 |
| causal-conv1d | 1.1.3 |
| numpy | 1.24.3 |
| pandas | 2.0.3 |

---

## 🧩 Environment Validation

Before running the project, please verify your environment.

### 1️⃣ Create `check_environment.py`

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Environment Validation Script for Traffic Flow Prediction Project
Based on Mamba Architecture
---------------------------------------------------------------
Verifies that all dependencies are correctly installed and accessible.
"""

import sys
import torch
import numpy as np
import pandas as pd

def check_environment():
    print("🔍 Checking environment configuration...\n")

    try:
        import mamba_ssm
        import causal_conv1d

        print("✅ All dependencies are installed correctly!\n")

        print(f"🐍 Python version: {sys.version.split()[0]}")
        print(f"🔥 PyTorch version: {torch.__version__}")
        print(f"🧩 mamba-ssm version: {getattr(mamba_ssm, '__version__', 'unknown')}")
        print(f"🔄 causal-conv1d version: {getattr(causal_conv1d, '__version__', 'unknown')}")
        print(f"🔢 numpy version: {np.__version__}")
        print(f"📊 pandas version: {pd.__version__}")

        # CUDA info
        print("\n💻 CUDA information:")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            device_id = torch.cuda.current_device()
            print(f"Current device ID: {device_id}")
            print(f"Device name: {torch.cuda.get_device_name(device_id)}")

        print("\n✅ Environment check completed successfully!")

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_environment()
```

2️⃣ Run Validation

```
python check_environment.py
```

🧠 Model Highlights

```
✅ Mamba-based architecture for efficient sequence modeling
✅ Multi-scale time series processing for dynamic temporal representation
✅ Spatio-temporal fusion to capture complex spatial correlations
✅ Supports CUDA acceleration
```# SC-_Mamba
