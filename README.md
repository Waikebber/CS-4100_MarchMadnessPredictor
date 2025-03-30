# CS-4100_MarchMadnessPredictor
---
## 🛠️ Environment Setup

To get started, create the conda environment using the provided `environment.yml` file. This will install all necessary dependencies for the project.

### Step 1: Create the Environment
```bash
conda env create -f environment.yml
```
- ⏳ *Note: This may take a few minutes to complete.*

### Step 2: Activate the Environment
```bash
conda activate marchmadness
```
---
## 📂 Data Setup

To set up the dataset for this project:

```bash
python data-setup.py <zip_file>
```

This will:
- Create a `./data` folder if it doesn't already exist
- Extract the contents of the zip file into `./data`

### Default Behavior
If no zip file is specified, the script will default to:
```python
ZIP_FILE = "./march-machine-learning-mania-2025.zip"
```

So you can simply run:
```bash
python data-setup.py
```