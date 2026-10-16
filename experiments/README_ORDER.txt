Recommended execution order (MATLAB R2019b+)

Easiest: everything in one go
  everything                     % master script (calls steps 1-9 below)

Individual steps (same order):
1. all_datasets                  % or nmi_q_log - main metrics, all datasets
2. ablation_all                  % ablation study
3. sensitivity_all               % = branches + label
   - or sensitivity_branches_all % B = 1..5
   - or sensitivity_label_all    % label ratios
4. separability_all
5. scalability_all               % train/infer time + peak memory
6. collect_results
7. statistical_tests             % Friedman + Wilcoxon on NMI
8. make_tables_main
9. make_tables_rel

Notes
- All settings: src/utils/get_user_settings.m
- DBLP is large (~317k nodes); 
- Use the commented 'quick test' dataset line in get_user_settings.m for faster debugging.
- scalability_all starts one fresh MATLAB process per seed (scalability_core.m).
  Use the same epoch settings (usr.epochs) as for all_datasets.
- Re-run an outlier timing seed by deleting results/scal_psaessl_<ds>_s<k>.mat.
