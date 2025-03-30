# CS-4100_MarchMadnessPredictor
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