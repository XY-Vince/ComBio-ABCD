import pandas as pd

# paths (adjust)
uni_path = "/analysis_output/univariate_tests_ADHD Unmedicated_vs_ADHD Stimulant.csv"
rf_path  = "/analysis_output/rf_importances_ADHD Unmedicated_vs_ADHD Stimulant.csv"
lr_path  = "/analysis_output/lr_coefficients_ADHD Unmedicated_vs_ADHD Stimulant.csv"

uni = pd.read_csv(uni_path)
rf  = pd.read_csv(rf_path)
lr  = pd.read_csv(lr_path)

# Top N
N = 5
# Univariate: sort by q (FDR) then effect size
uni_sorted = uni.sort_values(['q','cohens_d']).head(N)
uni_top = uni_sorted[['feature','mean_group1','mean_group2','cohens_d','p','q']]

# RF: top importances
rf_top = rf.sort_values('importance', ascending=False).head(N)[['feature','importance']]

# LR: top absolute coefficients
lr['abs_coef'] = lr['coef'].abs()
lr_top = lr.sort_values('abs_coef', ascending=False).head(N)[['feature','coef','p_value']]

print("Univariate top:\n", uni_top)
print("RF top:\n", rf_top)
print("LR top:\n", lr_top)