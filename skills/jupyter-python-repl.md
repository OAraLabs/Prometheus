---
name: jupyter-python-repl
description: Stateful Python execution for iterative exploration, data science, ML experimentation, and building up complex code step-by-step. Uses bash + python3 for all execution. Load this when the task involves exploration, iteration, or inspecting intermediate results.
version: 1.0.0
tags: [Python, REPL, Data-Science, Exploration, Iterative]
---

# Python REPL (Iterative Execution)

Stateful Python execution for iterative exploration, data science, ML, and building up complex code step-by-step. Prometheus runs all Python through `bash`.

## When to Use

| Approach | Use When |
|----------|----------|
| **This skill** | Iterative exploration, state across steps, data science, ML, "let me try this and check" |
| One-shot `bash` | Simple scripts, quick calculations, file processing |
| `file_write` + `bash` | Larger scripts that need to be saved and run |

**Rule of thumb:** If you would want a Jupyter notebook for the task, use this approach.

## Approaches

### 1. Inline Python via bash (simplest)

For quick one-liners and short explorations:

```bash
python3 -c "
import os
files = os.listdir('.')
print(f'Found {len(files)} files')
for f in sorted(files)[:10]:
    print(f'  {f}')
"
```

### 2. Python script files (for state persistence)

For iterative work where state needs to persist across steps, write a script and run it:

```bash
# Write the exploration script
cat > /tmp/explore.py << 'PYEOF'
import pandas as pd
import json

# Load data
df = pd.read_csv("data.csv")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.describe())

# Save intermediate results for next step
df.to_pickle("/tmp/explore_state.pkl")
PYEOF

python3 /tmp/explore.py
```

Then build on previous results:

```bash
cat > /tmp/explore_step2.py << 'PYEOF'
import pandas as pd

# Restore state from previous step
df = pd.read_pickle("/tmp/explore_state.pkl")

# Continue analysis
print(f"Null counts:\n{df.isnull().sum()}")
filtered = df[df['score'] > 0.8]
print(f"\nHigh scorers: {len(filtered)}")
filtered.to_pickle("/tmp/explore_state.pkl")
PYEOF

python3 /tmp/explore_step2.py
```

### 3. Python REPL via heredoc (multi-statement)

```bash
python3 << 'PYEOF'
import numpy as np

# Generate sample data
data = np.random.randn(1000)
print(f"Mean: {data.mean():.4f}")
print(f"Std:  {data.std():.4f}")
print(f"Min:  {data.min():.4f}")
print(f"Max:  {data.max():.4f}")

# Histogram (text-based)
hist, edges = np.histogram(data, bins=20)
max_count = max(hist)
for count, edge in zip(hist, edges):
    bar = '#' * int(40 * count / max_count)
    print(f"{edge:6.2f} | {bar}")
PYEOF
```

## Data Science Workflow

### Install packages

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Load and explore data

```bash
python3 << 'PYEOF'
import pandas as pd
import json

df = pd.read_csv("data.csv")
print("=== Shape ===")
print(df.shape)
print("\n=== Info ===")
print(df.dtypes)
print("\n=== Head ===")
print(df.head().to_string())
print("\n=== Describe ===")
print(df.describe().to_string())
PYEOF
```

### Generate plots (save to file)

```bash
python3 << 'PYEOF'
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df['column_a'].hist(ax=axes[0])
axes[0].set_title('Distribution of A')
df.plot.scatter(x='column_a', y='column_b', ax=axes[1])
axes[1].set_title('A vs B')
plt.tight_layout()
plt.savefig('/tmp/analysis_plot.png', dpi=150)
print("Plot saved to /tmp/analysis_plot.png")
PYEOF
```

## State Persistence Patterns

### Pickle for DataFrames and objects

```python
import pickle

# Save
with open('/tmp/state.pkl', 'wb') as f:
    pickle.dump({'df': df, 'model': model, 'params': params}, f)

# Load in next step
with open('/tmp/state.pkl', 'rb') as f:
    state = pickle.load(f)
    df = state['df']
```

### JSON for simple data

```python
import json

# Save
with open('/tmp/results.json', 'w') as f:
    json.dump({'accuracy': 0.95, 'features': ['a', 'b']}, f)

# Load
with open('/tmp/results.json') as f:
    results = json.load(f)
```

## Tips

1. Use `/tmp/` for scratch files and intermediate state.
2. For long-running computations, save intermediate results so you can resume.
3. Use `matplotlib.use('Agg')` for non-interactive plot generation.
4. Pipe large outputs through `head` or `tail` to keep output manageable.
5. For complex multi-step analyses, write each step as a separate script file for clarity.
6. Install packages once at the start; check with `python3 -c "import pandas; print(pandas.__version__)"`.
7. Use SENTINEL iteration caps if running exploratory loops -- limit retries to 3-5 attempts.
