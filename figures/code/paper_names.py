"""File names of the figures of the paper.

Each script that draws a figure of the paper writes it under the name that the LaTeX source of the paper includes
(Images/<name>). Intermediate figures, and the figures that the paper does not show, keep the names given by the
plotting functions.
"""
FIG1 = "fig1_openml_dataset_sizes.pdf"                    # fig1_openml_sizes.py
FIG2 = "fig2_variance_vs_test_size.pdf"                   # fig2_pedagogical.py
FIG3 = "fig3_dataset_breakdown.pdf"                       # diagram drawn by hand, no script
FIG4 = "fig4_gain_simulated_200_splits.pdf"               # paper_style_figs.py
FIG5_SIM = "fig5_gain_simulated.pdf"                      # fig5_common_axis.py: the three panels on one axis
FIG5_PCAM = "fig5_gain_pcam.pdf"
FIG5_YELP = "fig5_gain_yelp.pdf"
FIG6 = "fig6_redundancy_vs_gain.pdf"                      # fig6_redundancy_vs_gain.py
FIG7A = "fig7a_ranking_retrieval_yelp.pdf"                # regen_real_data_gain_figures.py --only fig5_yelp_aligned
FIG7B = "fig7b_test_scores_yelp.pdf"
FIGD1 = "figD1_fold_error_correlation_vs_gain.pdf"        # figD1_rho_delta.py
FIGG1 = "figG1_gain_hyperparameter_search.pdf"            # paper_style_figs.py
FIGH1 = "figH1_gain_frozen_vs_resampled_splits.pdf"       # paper_style_figs.py
FIGI1 = "figI1_gain_mccv_vs_repeated_kfold.pdf"           # regen_real_data_gain_figures.py --only figF1
FIGJ1 = "figJ1_gain_stratified_vs_unstratified.pdf"       # regen_real_data_gain_figures.py --only figK1
FIGK1 = "figK1_gain_by_metric.pdf"                        # regen_real_data_gain_figures.py --only figL1
FIGL1 = {30: "figL1_gain_small_budgets_ntr30.pdf",        # paper_style_figs.py, one file per panel
         100: "figL1_gain_small_budgets_ntr100.pdf"}
FIGO1A = "figO1a_ranking_equivalent_gain.pdf"             # code/ranking.py
FIGO1B = "figO1b_ranking_retrieval_curves.pdf"

# The panels of Figure 5, each on its own axis, as the gain scripts draw them before fig5_common_axis.py.
FIG5_SIM_OWN_AXIS = "fig5_gain_simulated_own_axis.pdf"    # paper_style_figs.py
FIG5_PCAM_OWN_AXIS = "fig5_gain_pcam_own_axis.pdf"        # regen_real_data_gain_figures.py --only fig5_pcam
FIG5_YELP_OWN_AXIS = "fig5_gain_yelp_own_axis.pdf"        # regen_real_data_gain_figures.py --only fig5_yelp_aligned
