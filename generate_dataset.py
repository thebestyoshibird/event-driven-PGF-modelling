from sim_sweep_v2 import *

# STATIC MODEL PARAMETERS
MONTH_LEN = 30.44      # days/month (~365/12)
N_MONTHS = 36           # 36-month observation window
DEFAULT_T = 365 * 3

# RUN MODEL TO GENERATE DATA

if __name__ == "__main__":

    P_h_example = [0.0025, 0.005, 0.0075, 0.01, 0.0125]
    P_e_example = [0.0025, 0.005, 0.0075, 0.01, 0.0125]
    P_u_example = [0.0025, 0.005, 0.0075, 0.01, 0.0125]

    summary_df, timeseries = run_sweep(
        P_h_example, P_e_example, P_u_example,
        numsims=10, size_pop=20000, T=DEFAULT_T,
        record_timeseries=True, fix_known_bugs=True,
        n_workers=None, base_seed=None,
    )

    save_sweep(summary_df, timeseries, out_dir="sweep_results", tag="pu_pe_ph")
    print(summary_df.head())
