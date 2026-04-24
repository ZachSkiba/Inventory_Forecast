# src/helpers.py (or just helpers.py)

import logging
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

def prophet_cv_search(train_df, param_grid, initial, period, horizon):
    logging.getLogger('prophet').setLevel(logging.ERROR)
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    
    results = []
    total = len(param_grid)
    for i, params in enumerate(param_grid):
        try:
            m = Prophet(
                changepoint_prior_scale = params['changepoint_prior_scale'],
                seasonality_prior_scale = params.get('seasonality_prior_scale', 1.0),     
                seasonality_mode        = params['seasonality_mode'],
                changepoint_range       = 0.8,     
                holidays_prior_scale    = params.get('holidays_prior_scale', 10.0),    
                yearly_seasonality      = True,
                weekly_seasonality      = False,
                daily_seasonality       = False,
                interval_width          = 0.95,
                uncertainty_samples     = 200
            )
            m.add_seasonality(
                name          = 'quarterly',
                period        = 91.25,
                fourier_order = 3,
                mode          = params['seasonality_mode']
            )
            m.fit(train_df)
            df_cv = cross_validation(
                m, initial=initial, period=period,
                horizon=horizon, disable_tqdm=True
            )
            df_perf = performance_metrics(df_cv, rolling_window=1)
            rmse = df_perf['rmse'].mean()
            mape = df_perf['mape'].mean() * 100
            results.append({**params, 'rmse': round(rmse, 2), 'mape': round(mape, 2)})
            print(f'  [{i+1}/{total}] cps={params["changepoint_prior_scale"]} '
                  f'mode={params["seasonality_mode"]} '
                  f'RMSE=${rmse:,.2f} MAPE={mape:.1f}%')
        except Exception as e:
            print(f'  [{i+1}/{total}] FAILED: {e}')
            continue
    return pd.DataFrame(results).sort_values('rmse').reset_index(drop=True)