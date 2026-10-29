import numpy as np
import pandas as pd
from arch import arch_model
from main import CONFIG, stock_getter

print("⚡ Génération du modèle Custom 'EGARCH-Skewt Champion'...")

# 1. Chargement des rendements
returns = stock_getter(
    CONFIG["ticker"], start=CONFIG["start_date"], end=CONFIG["end_date"]
)
window_size = CONFIG["window_size"]
refit_freq = CONFIG["refit_freq"]
alpha_var = CONFIG["alpha_var"]
alpha_es = CONFIG["alpha_es"]

n_obs = len(returns)
test_index = returns.index[window_size:]
n_test = len(test_index)

var_arr = np.empty(n_test)
es_arr = np.empty(n_test)
vol_arr = np.empty(n_test)

# 2. Backtest glissant EGARCH(1,1) - Skew-t
last_params = None

for i in range(window_size, n_obs, refit_freq):
    train_returns = returns.iloc[i - window_size : i]
    end_idx = min(i + refit_freq, n_obs)
    horizon_len = end_idx - i
    test_block = returns.iloc[i - window_size : end_idx]

    try:
        # Modèle EGARCH(1,1) avec distribution Skew-t
        model = arch_model(
            train_returns * 100,
            p=1,
            o=1,
            q=1,
            vol="EGARCH",
            dist="skewt",
            mean="Zero",
        )
        fit = model.fit(disp="off", show_warning=False)
        last_params = fit.params
    except Exception:
        if last_params is not None:
            fit_params = last_params
        else:
            fit_params = fit.params

    # Ajustement sur le bloc courant
    current_model = arch_model(
        test_block * 100, p=1, o=1, q=1, vol="EGARCH", dist="skewt", mean="Zero"
    )
    fitted = current_model.fix(last_params)

    cond_vol = fitted.conditional_volatility.iloc[-horizon_len:] / 100

    target_start = i - window_size
    target_end = target_start + horizon_len

    vol_arr[target_start:target_end] = cond_vol.values

    # Paramètres de forme
    eta = last_params.get("eta", last_params.get("nu", 6.0))
    lam = last_params.get("lambda", -0.1)
    params_shape = [eta, lam]

    # Quantiles VaR & ES
    q_var = current_model.distribution.ppf(alpha_var, params_shape)
    var_arr[target_start:target_end] = (cond_vol * np.abs(q_var)).values

    # ES numérique (intégration robuste)
    u = (np.arange(2000) + 0.5) / 2000 * alpha_es
    q_es_std = -np.mean(current_model.distribution.ppf(u, params_shape))
    es_arr[target_start:target_end] = (cond_vol * q_es_std).values

# 3. Exportation des fichiers CSV prêts pour l'UI Streamlit
df_var = pd.Series(var_arr, index=test_index, name="VaR")
df_es = pd.Series(es_arr, index=test_index, name="ES")
df_vol = pd.Series(vol_arr, index=test_index, name="Vol")

df_var.to_csv("var_champion.csv")
df_es.to_csv("es_champion.csv")
df_vol.to_csv("vol_champion.csv")

print(
    "🏆 Fichiers 'var_champion.csv', 'es_champion.csv' et 'vol_champion.csv' générés avec succès !"
)