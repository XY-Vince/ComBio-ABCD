# Quick Start Guide

Get up and running with the ADHD Digital Phenotype Analysis Pipeline in 5 minutes!

## ✅ Prerequisites

- Python 3.8 or higher
- Fitbit data Excel file
- medication data Excel file

## 🚀 Step-by-Step Setup

### Step 1: Install Python Packages

```bash
pip install -r requirements.txt
```

### Step 2: Update Configuration

Open `config.py` and update these lines:

```python
# Line 13-15: Update data paths
DATA_DIR = "/Users/guozhenghui/Desktop/WXY/ComBio/ABCD"  # data directory
FITBIT_FILE = os.path.join(DATA_DIR, "your_fitbit_file.xlsx")  # Fitbit file
MEDICATION_FILE = os.path.join(DATA_DIR, "medsy01.xlsx")  # medication file
```

### Step 3: Verify Data Structure

**Required columns in Fitbit data:**
- `subjectkey` - Subject identifier
- Fitbit features (automatically detected from column names)
- Optional: `adhd_diagnosis` - ADHD diagnosis indicator

**Required columns in Medication data:**
- `subjectkey` - Subject identifier  
- `med1_rxnorm_p` through `med5_rxnorm_p` - Medication entries

### Step 4: Run the Pipeline

```bash
python main_pipeline.py
```

That's it! The pipeline will:
1. Load and merge data ✓
2. Remove covariate effects ✓
3. Train predictive models ✓
4. Create visualizations ✓
5. Generate reports ✓

## 📊 Check Results

Results are saved in `analysis_output/` directory:

```
analysis_output/
├──           # Start here!
├── pipeline.log               # Detailed execution log
├── pca_2d_plot.png           # Visualization of groups
├── model_*_results.png       # Model performance plots
├── residualized_data.csv     # Processed data
└── pipeline_results.json     # All results (machine-readable)
```

## 🎯 Understanding Results

### 1. Open ``

This human-readable report shows:
- How many subjects in each group
- Model accuracy scores
- Which features are most important

### 2. View `pca_2d_plot.png`

This visualization shows:
- Whether groups separate into distinct clusters
- How much overlap exists between groups
- The "digital phenotype" landscape

### 3. Check Model Performance

Look for ROC-AUC scores in the report:
- **> 0.7** = Good separation between groups
- **> 0.8** = Excellent separation
- **< 0.6** = Groups are very similar

## ⚡ Advanced Options

### Run with Hyperparameter Tuning

```bash
python main_pipeline.py --tune
```
Takes longer but may improve results.

### Skip Validation (Faster)

```bash
python main_pipeline.py --skip-validation
```

### Specify Custom Files

```bash
python main_pipeline.py --fitbit-file /path/to/fitbit.xlsx --med-file /path/to/med.xlsx
```

## 🔧 Common Adjustments

### Change Which Groups to Compare

Edit `config.py` line ~100:

```python
COMPARISON_PAIRS = [
    ('control', 'unmedicated_adhd'),
    ('control', 'stimulant_adhd'),
    ('unmedicated_adhd', 'stimulant_adhd'),
    # Add own comparisons here
]
```

### Adjust Covariates

Edit `config.py` line ~44:

```python
COVARIATES = [
    'interview_age',
    'sex',
    'family_income',
    'race_eth_cat',
    # Add or remove covariates here
]
```

### Change Model Parameters

Edit `config.py` lines ~120-135 for Logistic Regression and Random Forest parameters.

## ❗ Troubleshooting

### Error: "File not found"
→ Check file paths in `config.py` line 13-15

### Error: "Column not found"  
→ Check data has required columns (see Step 3)

### Error: "Insufficient control subjects"
→ Need at least 20 healthy controls in data

### Warning: "High missing rate"
→ Normal if some Fitbit features have missing data

### Poor Model Performance (AUC < 0.6)
→ Try `--tune` option or check if groups are truly different

## 📞 Need Help?

1. Check `pipeline.log` for detailed error messages
2. Review `README.md` for comprehensive documentation
3. Contact: your.email@institution.edu

## 🎓 Next Steps

Once you've run the pipeline successfully:

1. **Examine Top Features**: Look at `lr_coefficients_*.csv` files
2. **Interpret PCA**: Understand what PC1 and PC2 represent
3. **Compare Groups**: See which features distinguish groups
4. **Customize**: Modify config.py for specific research questions

## 📈 Example Timeline

| Task | Time |
|------|------|
| Setup | 2 minutes |
| Basic run | 5-10 minutes |
| With tuning | 30-60 minutes |
| Large dataset | 1-2 hours |

---

**Happy Analyzing!** 🎉

For detailed documentation, see `README.md`




