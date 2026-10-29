import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from arch.utility.exceptions import StartingValueWarning
from scipy.stats import chi2

warnings.filterwarnings("ignore", category=StartingValueWarning)

CONFIG = {
    "ticker": "^GSPC",
    "start_date": "2001-01-01",
    "end_date": None,
    "window_size": 1500,
    "refit_freq": 20,
    "alpha_var": 0.01,
    "alpha_es": 0.025,
    "lags_dq": 4,
    "n_boot": 10000,
}


def stock_getter(ticker="AAPL", start="2001-01-01", end=None):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df_prices = df["Close"][[ticker]].copy()
        df_prices.columns = ["Close"]
    else:
        df_prices = df[["Close"]].copy()

    df_prices["log_return"] = np.log(
        df_prices["Close"] / df_prices["Close"].shift(1)
    )
    return df_prices["log_return"].dropna()


def es_standardized(dist, params, alpha, n=2000):
    try:
        u = (np.arange(n) + 0.5) / n * alpha
        return -np.mean(dist.ppf(u, params))
    except Exception:
        q = dist.ppf(alpha, params)
        return -q * 1.5


def dq_test(returns, var_series, alpha=0.01, lags=4):
    hits = (returns < -var_series).astype(float) - alpha
    n = len(hits)
    if n <= lags + 10:
        return np.nan, np.nan

    cols = [np.ones(n - lags), var_series.iloc[lags:].values]
    for lag in range(1, lags + 1):
        cols.append(hits.iloc[lags - lag : n - lag].values)

    X = np.column_stack(cols)
    Y = hits.iloc[lags:].values

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        return np.nan, np.nan

    norms = np.linalg.norm(X, axis=0)
    safe_norms = np.where(norms > 1e-10, norms, 1.0)

    with np.errstate(all="ignore"):
        X = X / safe_norms
        rank = np.linalg.matrix_rank(X)
        if rank == 0:
            return np.nan, np.nan
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        preds = X @ beta
        stat = np.sum(preds**2) / (alpha * (1.0 - alpha))

    pval = 1.0 - chi2.cdf(stat, df=rank)
    return float(stat), float(pval)


def mcneil_frey_es_bootstrap(
    test_returns, var_series, es_var_series, vol_series, n_boot=10000
):
    exceedances_mask = test_returns < -var_series
    if exceedances_mask.sum() < 3:
        return np.nan, np.nan

    r_tail = test_returns[exceedances_mask]
    es_tail = es_var_series[exceedances_mask]
    sigma_tail = vol_series[exceedances_mask]

    tail_residuals = (-r_tail - es_tail) / sigma_tail
    mean_obs = tail_residuals.mean()

    rng = np.random.default_rng(42)
    centered_resid = tail_residuals - mean_obs
    boot_samples = rng.choice(
        centered_resid, size=(n_boot, len(tail_residuals)), replace=True
    )
    boot_means = boot_samples.mean(axis=1)

    p_val = np.mean(boot_means >= mean_obs)
    if p_val == 0.0:
        p_val = 1.0 / n_boot

    return float(p_val), float(mean_obs)


def acerbi_szekely_z1_test(returns, var_series, es_series, n_boot=10000):
    mask = returns < -var_series
    if mask.sum() < 5:
        return np.nan, np.nan

    ratio = (-returns[mask] / es_series[mask]).values
    z1_obs = ratio.mean() - 1.0

    rng = np.random.default_rng(42)
    centered = ratio - ratio.mean()
    boot = rng.choice(centered, size=(n_boot, len(ratio)), replace=True).mean(
        axis=1
    )

    p_val = np.mean(boot >= z1_obs)
    if p_val == 0.0:
        p_val = 1.0 / n_boot

    return float(z1_obs), float(p_val)


def basel_penalty(k_exc):
    if k_exc <= 4:
        return "VERTE", 3.00
    elif k_exc == 5:
        return "JAUNE", 3.40
    elif k_exc == 6:
        return "JAUNE", 3.50
    elif k_exc == 7:
        return "JAUNE", 3.65
    elif k_exc == 8:
        return "JAUNE", 3.75
    elif k_exc == 9:
        return "JAUNE", 3.85
    else:
        return "ROUGE", 4.00


def champion_key(outputs, alpha=0.01):
    ok = [
        k
        for k, r in outputs.items()
        if isinstance(r["kupiec_pval"], (float, int))
        and not np.isnan(r["kupiec_pval"])
        and r["kupiec_pval"] > 0.05
        and isinstance(r["dq_pval"], (float, int))
        and not np.isnan(r["dq_pval"])
        and r["dq_pval"] > 0.05
    ]
    pool = ok or list(outputs)
    return min(
        pool,
        key=lambda k: abs(
            (outputs[k]["exceptions"] / outputs[k]["obs_tested"]) - alpha
        ),
    )


def run_backtest(
    returns,
    model_type="gjr_skewt",
    window_size=1500,
    refit_freq=20,
    alpha_var=0.01,
    alpha_es=0.025,
    custom_data=None,
):
    n_obs = len(returns)
    test_index = returns.index[window_size:]
    n_test = len(test_index)

    var_series_arr = np.empty(n_test)
    var_frtb_arr = np.empty(n_test)
    es_var_arr = np.empty(n_test)
    es_frtb_arr = np.empty(n_test)
    vol_series_arr = np.empty(n_test)

    eta_hist, lambda_hist = [], []
    refit_dates = []
    conv_warnings = 0

    if model_type == "custom":
        if custom_data is None or not isinstance(custom_data, dict):
            raise ValueError("custom_data requis pour le modèle custom.")

        var_c = pd.Series(custom_data["var"]).reindex(test_index).ffill().bfill()
        es_c = pd.Series(custom_data["es"]).reindex(test_index).ffill().bfill()
        vol_c = pd.Series(custom_data["vol"]).reindex(test_index).ffill().bfill()

        var_series_arr = np.abs(var_c.values)
        var_frtb_arr = np.abs(var_c.values)
        es_var_arr = np.abs(es_c.values)
        es_frtb_arr = np.abs(es_c.values)
        vol_series_arr = np.abs(vol_c.values)

        refit_dates = list(test_index)

    elif model_type == "historical":
        for idx_i, i in enumerate(range(window_size, n_obs)):
            train_returns = returns.iloc[i - window_size : i]
            q_var_val = np.quantile(train_returns, alpha_var)
            q_var_frtb_val = np.quantile(train_returns, alpha_es)
            q_es_frtb_val = np.quantile(train_returns, alpha_es)

            es_var_val = train_returns[train_returns <= q_var_val].mean()
            es_frtb_val = train_returns[train_returns <= q_es_frtb_val].mean()
            vol_val = train_returns.std()

            var_series_arr[idx_i] = -q_var_val
            var_frtb_arr[idx_i] = -q_var_frtb_val
            es_var_arr[idx_i] = -es_var_val
            es_frtb_arr[idx_i] = -es_frtb_val
            vol_series_arr[idx_i] = vol_val
            refit_dates.append(returns.index[i])

    else:
        last_valid_params = None
        for i in range(window_size, n_obs, refit_freq):
            train_returns = returns.iloc[i - window_size : i]
            end_idx = min(i + refit_freq, n_obs)
            horizon_len = end_idx - i
            test_block = returns.iloc[i - window_size : end_idx]
            refit_dates.append(returns.index[i])

            p, o, q = (1, 1, 1) if model_type == "gjr_skewt" else (1, 0, 1)
            dist_type = "skewt" if model_type == "gjr_skewt" else "studentst"

            try:
                model = arch_model(
                    train_returns * 100, p=p, o=o, q=q, dist=dist_type, mean="Zero"
                )
                model_fit = model.fit(disp="off", show_warning=False)

                if model_fit.convergence_flag != 0:
                    conv_warnings += 1

                current_params = model_fit.params
                last_valid_params = current_params
            except Exception:
                conv_warnings += 1
                if last_valid_params is not None:
                    current_params = last_valid_params
                else:
                    current_params = pd.Series(
                        [0.05, 0.05, 0.10, 0.80, 5.0]
                        if dist_type == "studentst"
                        else [0.05, 0.05, 0.10, 0.80, 5.0, 0.0],
                        index=[
                            "omega",
                            "alpha[1]",
                            "beta[1]",
                            "gamma[1]",
                            "nu",
                            "lambda",
                        ][: 5 + (1 if dist_type == "skewt" else 0)],
                    )

            current_model = arch_model(
                test_block * 100, p=p, o=o, q=q, dist=dist_type, mean="Zero"
            )
            fitted = current_model.fix(current_params)
            cond_vol = fitted.conditional_volatility.iloc[-horizon_len:] / 100

            target_idx_start = i - window_size
            target_idx_end = target_idx_start + horizon_len
            vol_series_arr[target_idx_start:target_idx_end] = cond_vol.values

            if dist_type == "skewt":
                eta_val = current_params.get(
                    "eta", current_params.get("nu", 5.0)
                )
                lambda_val = current_params.get("lambda", 0.0)
                eta_hist.append(eta_val)
                lambda_hist.append(lambda_val)
                params_shape = [eta_val, lambda_val]
            else:
                nu_val = current_params.get("nu", 5.0)
                eta_hist.append(nu_val)
                params_shape = [nu_val]

            q_var = current_model.distribution.ppf(alpha_var, params_shape)
            var_series_arr[target_idx_start:target_idx_end] = -(
                cond_vol * q_var
            ).values

            q_var_frtb = current_model.distribution.ppf(alpha_es, params_shape)
            var_frtb_arr[target_idx_start:target_idx_end] = -(
                cond_vol * q_var_frtb
            ).values

            es_std_var = es_standardized(
                current_model.distribution, params_shape, alpha_var
            )
            es_var_arr[target_idx_start:target_idx_end] = (
                cond_vol * es_std_var
            ).values

            es_std_frtb = es_standardized(
                current_model.distribution, params_shape, alpha_es
            )
            es_frtb_arr[target_idx_start:target_idx_end] = (
                cond_vol * es_std_frtb
            ).values

    var_series = pd.Series(var_series_arr, index=test_index)
    var_frtb_series = pd.Series(var_frtb_arr, index=test_index)
    es_var_series = pd.Series(es_var_arr, index=test_index)
    es_frtb_series = pd.Series(es_frtb_arr, index=test_index)
    vol_series = pd.Series(vol_series_arr, index=test_index)

    test_returns = returns.iloc[window_size:]
    exceptions = test_returns < -var_series
    x = int(exceptions.sum())
    N = len(test_returns)

    x_N = x / N if N > 0 else 0
    ll_h0 = (N - x) * np.log(1.0 - alpha_var) + (
        x * np.log(alpha_var) if x > 0 else 0
    )
    ll_h1 = (
        (N - x) * np.log(1.0 - x_N) + x * np.log(x_N) if 0 < x < N else 0
    )
    LR_pof = max(0.0, -2.0 * (ll_h0 - ll_h1))
    pval_pof = float(1.0 - chi2.cdf(LR_pof, df=1))

    exc_int = exceptions.astype(int).values
    T00 = T01 = T10 = T11 = 0
    for t in range(1, len(exc_int)):
        if exc_int[t - 1] == 0 and exc_int[t] == 0:
            T00 += 1
        elif exc_int[t - 1] == 0 and exc_int[t] == 1:
            T01 += 1
        elif exc_int[t - 1] == 1 and exc_int[t] == 0:
            T10 += 1
        elif exc_int[t - 1] == 1 and exc_int[t] == 1:
            T11 += 1

    pi_0 = T01 / (T00 + T01) if (T00 + T01) > 0 else 0
    pi_1 = T11 / (T10 + T11) if (T10 + T11) > 0 else 0
    pi_hat = (
        (T01 + T11) / (T00 + T01 + T10 + T11) if (T00 + T01 + T10 + T11) > 0 else 0
    )

    ll_ind_null = (
        (T00 + T10) * np.log(1.0 - pi_hat) + (T01 + T11) * np.log(pi_hat)
        if 0 < pi_hat < 1
        else 0
    )
    ll_ind_alt = 0
    if T00 > 0 and (1.0 - pi_0) > 0:
        ll_ind_alt += T00 * np.log(1.0 - pi_0)
    if T01 > 0 and pi_0 > 0:
        ll_ind_alt += T01 * np.log(pi_0)
    if T10 > 0 and (1.0 - pi_1) > 0:
        ll_ind_alt += T10 * np.log(1.0 - pi_1)
    if T11 > 0 and pi_1 > 0:
        ll_ind_alt += T11 * np.log(pi_1)

    LR_ind = max(0.0, -2.0 * (ll_ind_null - ll_ind_alt))
    pval_ind = float(1.0 - chi2.cdf(LR_ind, df=1))
    LR_cc = LR_pof + LR_ind
    pval_cc = float(1.0 - chi2.cdf(LR_cc, df=2))

    dq_stat, dq_pval = dq_test(
        test_returns, var_series, alpha=alpha_var, lags=CONFIG["lags_dq"]
    )
    es_pval, mean_excess = mcneil_frey_es_bootstrap(
        test_returns,
        var_series,
        es_var_series,
        vol_series,
        n_boot=CONFIG["n_boot"],
    )
    as_z1_stat, as_z1_pval = acerbi_szekely_z1_test(
        test_returns, var_frtb_series, es_frtb_series, n_boot=CONFIG["n_boot"]
    )

    rolling_250 = exceptions.astype(int).rolling(250).sum().dropna()
    max_basel_exc = int(rolling_250.max()) if len(rolling_250) > 0 else 0
    pct_red = (
        (rolling_250 >= 10).mean() * 100 if len(rolling_250) > 0 else 0.0
    )
    current_zone = (
        basel_penalty(int(rolling_250.iloc[-1]))[0]
        if len(rolling_250) > 0
        else "VERTE"
    )

    years = test_returns.index.year.astype(str)
    exc_2008 = (
        int(exceptions.loc[years == "2008"].sum()) if "2008" in years.values else 0
    )
    exc_2020 = (
        int(exceptions.loc[years == "2020"].sum()) if "2020" in years.values else 0
    )
    exc_2022 = (
        int(exceptions.loc[years == "2022"].sum()) if "2022" in years.values else 0
    )

    return {
        "model": model_type,
        "obs_tested": N,
        "exceptions": x,
        "exc_rate": (x / N) * 100,
        "kupiec_pval": pval_pof,
        "christof_ind_pval": pval_ind,
        "christof_cc_pval": pval_cc,
        "t11_count": T11,
        "dq_pval": dq_pval,
        "es_mf_pval": es_pval,
        "acerbi_z1_stat": as_z1_stat,
        "acerbi_z1_pval": as_z1_pval,
        "tail_resid_mean": mean_excess,
        "conv_warnings": conv_warnings,
        "basel_zone_actu": current_zone,
        "basel_pct_red": pct_red,
        "basel_max_250j": max_basel_exc,
        "exc_2008_crisis": exc_2008,
        "exc_2020_covid": exc_2020,
        "exc_2022_rates": exc_2022,
        "eta_min_max": (
            (min(eta_hist), max(eta_hist)) if eta_hist else (np.nan, np.nan)
        ),
        "lambda_min_max": (
            (min(lambda_hist), max(lambda_hist))
            if lambda_hist
            else (np.nan, np.nan)
        ),
        "var_series": var_series,
        "es_frtb_series": es_frtb_series,
        "test_returns": test_returns,
        "rolling_250": rolling_250,
        "eta_hist": eta_hist,
        "lambda_hist": lambda_hist,
        "refit_dates": refit_dates,
    }


def plot_champion_results(champion):
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 11), gridspec_kw={"height_ratios": [2.5, 1, 1]}
    )

    ax1.plot(
        champion["test_returns"].index,
        champion["test_returns"],
        label="Rendements Réels",
        color="gray",
        alpha=0.4,
        lw=0.7,
    )
    ax1.plot(
        champion["var_series"].index,
        -champion["var_series"],
        label="VaR 99%",
        color="red",
        lw=1.2,
    )
    ax1.plot(
        champion["es_frtb_series"].index,
        -champion["es_frtb_series"],
        label="ES 97.5% (Calibré FRTB)",
        color="orange",
        lw=1.2,
        linestyle="--",
    )
    exceptions_mask = champion["test_returns"] < -champion["var_series"]
    ax1.scatter(
        champion["test_returns"][exceptions_mask].index,
        champion["test_returns"][exceptions_mask],
        color="darkred",
        s=20,
        zorder=5,
        label="Exceptions VaR 99%",
    )
    ax1.set_title(f"1. Backtest VaR 99% & Expected Shortfall 97.5% - Modèle: {champion['model']}")
    ax1.legend(loc="lower left")
    ax1.grid(True, alpha=0.3)

    ax2.plot(
        champion["rolling_250"].index,
        champion["rolling_250"],
        color="black",
        lw=1.2,
        label="Exceptions (Rolling 250j)",
    )
    ax2.axhline(
        4.5, color="green", linestyle="--", linewidth=1, label="Zone Verte (<=4)"
    )
    ax2.axhline(
        9.5, color="orange", linestyle="--", linewidth=1, label="Zone Jaune (5-9)"
    )
    ax2.set_ylabel("Nb Exceptions")
    ax2.set_title("2. Profil de Bâle (Fenêtre glissante 250 jours)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    refit_dates = champion["refit_dates"]
    is_skew = champion["model"] == "gjr_skewt"
    color1, color2 = "tab:purple", "teal"

    ax3.set_title(
        "3. Stabilité temporelle ("
        + ("Skew-t de Hansen" if is_skew else "Student-t")
        + ")"
    )
    label_eta = "Degrés de liberté (η)" if is_skew else "Degrés de liberté (ν)"

    if champion["model"] not in ["historical", "custom"] and champion["eta_hist"]:
        ax3.plot(
            refit_dates,
            champion["eta_hist"],
            color=color1,
            lw=1.2,
            label=label_eta,
        )
        ax3.set_ylabel(label_eta, color=color1)
        ax3.tick_params(axis="y", labelcolor=color1)
        ax3.grid(True, alpha=0.3)

        if is_skew:
            ax3_twin = ax3.twinx()
            ax3_twin.plot(
                refit_dates,
                champion["lambda_hist"],
                color=color2,
                lw=1.2,
                linestyle=":",
                label="Asymétrie (λ)",
            )
            ax3_twin.set_ylabel("Paramètre λ (Asymétrie)", color=color2)
            ax3_twin.tick_params(axis="y", labelcolor=color2)

            h1, l1 = ax3.get_legend_handles_labels()
            h2, l2 = ax3_twin.get_legend_handles_labels()
            ax3.legend(h1 + h2, l1 + l2, loc="upper left")
        else:
            ax3.legend(loc="upper left")
    else:
        ax3.text(
            0.5,
            0.5,
            "Paramètres de queue non applicables",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax3.transAxes,
        )

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    returns = stock_getter(
        CONFIG["ticker"], start=CONFIG["start_date"], end=CONFIG["end_date"]
    )

    models_to_test = ["historical", "garch_t", "gjr_skewt"]
    benchmark_results = []
    dict_outputs = {}

    for m in models_to_test:
        res = run_backtest(
            returns,
            model_type=m,
            window_size=CONFIG["window_size"],
            refit_freq=CONFIG["refit_freq"],
            alpha_var=CONFIG["alpha_var"],
            alpha_es=CONFIG["alpha_es"],
        )
        dict_outputs[m] = res

        clean_res = {
            k: v
            for k, v in res.items()
            if k
            not in [
                "var_series",
                "es_frtb_series",
                "test_returns",
                "rolling_250",
                "eta_hist",
                "lambda_hist",
                "refit_dates",
            ]
        }
        benchmark_results.append(clean_res)

    df_summary = pd.DataFrame(benchmark_results)

    print("\n" + "=" * 80)
    print(" TABLEAU COMPARATIF D'AUDIT DU RISQUE ")
    print("=" * 80)
    print(df_summary.set_index("model").T.to_string())

    best_model_key = champion_key(dict_outputs, alpha=CONFIG["alpha_var"])
    champion = dict_outputs[best_model_key]

    print(f"\nModèle sélectionné : {champion['model'].upper()}")

    fig = plot_champion_results(champion)
    plt.show()