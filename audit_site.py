import datetime
import html
import re
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from streamlit.web import cli as stcli

from main import CONFIG, run_backtest, stock_getter

STANDARD_MODELS = {
    "historical": "Historique (HS)",
    "garch_t": "GARCH-t",
    "gjr_skewt": "GJR-GARCH (Skew-t)",
}

METRIC_LABELS = {
    "basel_zone_actu": "Zone de Bâle (actuelle)",
    "exceptions": "Exceptions VaR",
    "exc_rate": "Taux d'exceptions (%)",
    "obs_tested": "Observations testées",
    "kupiec_pval": "Kupiec POF · p-value",
    "christof_cc_pval": "Christoffersen CC · p-value",
    "dq_pval": "Dynamic Quantile · p-value",
    "christof_ind_pval": "Christoffersen IND · p-value",
    "es_mf_pval": "ES McNeil-Frey · p-value",
    "acerbi_z1_stat": "Acerbi-Szekely Z1 · statistique",
    "acerbi_z1_pval": "Acerbi-Szekely Z1 · p-value",
    "tail_resid_mean": "Moyenne des résidus de queue",
    "conv_warnings": "Alertes de convergence",
    "exc_2008_crisis": "Exceptions · crise 2008",
}

MODEL_COLORS = []
C = {}


def _norm_hex(h, default):
    try:
        h = str(h).strip()
        if h.startswith("#") and len(h) == 4:
            h = "#" + "".join(ch * 2 for ch in h[1:])
        if h.startswith("#") and len(h) == 7:
            int(h[1:], 16)
            return h.upper()
    except (ValueError, TypeError):
        pass
    return default


def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _hex(rgb):
    return "#%02X%02X%02X" % tuple(int(round(max(0, min(255, x)))) for x in rgb)


def _mix(a, b, t):
    ra, rb = _rgb(a), _rgb(b)
    return _hex([ra[i] * (1 - t) + rb[i] * t for i in range(3)])


def _lum(h):
    def ch(v):
        v = v / 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = _rgb(h)
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def _file_theme(section_name="theme"):
    from pathlib import Path
    root = Path(__file__).resolve().parent
    for cand in (root / ".streamlit" / "config.toml", root / "config.toml", root / "Config.toml"):
        try:
            if not cand.is_file():
                continue
            theme, current_section = {}, None
            for raw in cand.read_text(encoding="utf-8").splitlines():
                line = raw.split("#", 1)[0].strip() if '"#' not in raw else raw.strip()
                m = re.match(r"^\[([^\]]+)\]$", line)
                if m:
                    current_section = m.group(1).strip()
                    continue
                m = re.match(r'^(\w+)\s*=\s*"([^"]*)"', line)
                if m and current_section == section_name:
                    theme[m.group(1)] = m.group(2)
            if theme:
                return theme
        except Exception:
            continue
    return {}


def _series_colors(primary, ink, panel, n=6):
    import colorsys
    r, g, b = [x / 255 for x in _rgb(primary)]
    h, _, sat = colorsys.rgb_to_hls(r, g, b)
    sat = max(sat, 0.55)
    light = 0.72 if _lum(panel) < 0.4 else 0.38
    out = [ink]
    for k in range(n - 1):
        rr, gg, bb = colorsys.hls_to_rgb((h + k / (n - 1)) % 1.0, light, sat)
        out.append(_hex((rr * 255, gg * 255, bb * 255)))
    return out


def _theme_opt(name, default):
    try:
        v = st.get_option(f"theme.{name}")
    except Exception:
        v = None
    if not v:
        v = _file_theme().get(name)
    return _norm_hex(v, default) if v else default


def apply_theme():
    mode = st.session_state.get("theme_mode", "dark")
    section_name = f"theme_{mode}"
    cfg_theme = _file_theme(section_name)

    if cfg_theme:
        bg = _norm_hex(cfg_theme.get("backgroundColor"), "#F2F4F7")
        panel = _norm_hex(cfg_theme.get("secondaryBackgroundColor"), "#FFFFFF")
        ink = _norm_hex(cfg_theme.get("textColor"), "#121B2B")
        primary = _norm_hex(cfg_theme.get("primaryColor"), "#0B1F3A")
        primary_set = True
    else:
        bg = _theme_opt("backgroundColor", "#F2F4F7")
        panel = _theme_opt("secondaryBackgroundColor", "#FFFFFF")
        ink = _theme_opt("textColor", "#121B2B")
        primary_set = bool(st.get_option("theme.primaryColor") or _file_theme().get("primaryColor"))
        primary = _theme_opt("primaryColor", "#0B1F3A")

    dark_panel = _lum(panel) < 0.4
    side = primary if _lum(primary) < 0.02 else min((ink, primary), key=_lum)
    if _lum(side) > 0.08:
        side = _mix(primary, "#000000", 0.7)
    if _lum(ink) > 0.5:
        side = _mix(bg, "#000000", 0.45)
    if _lum(primary) > 0.15:
        side_accent = primary
    else:
        side_accent = _mix(primary, "#FFFFFF", 0.5) if primary_set else "#6FA8E8"

    muted = _mix(ink, panel, 0.3)
    line = _mix(panel, ink, 0.18)
    line_strong = _mix(panel, ink, 0.42)
    grid = _mix(panel, ink, 0.10)
    hover = _mix(panel, ink, 0.05)
    soft = _mix(panel, ink, 0.03)

    if dark_panel:
        pos, warn, neg = "#8FF0B5", "#FFD28A", "#FFA39E"
    else:
        pos, warn, neg = "#16794A", "#A8620A", "#B3261E"

    blue = primary if primary_set else "#1F4E8C"
    phover = _mix(primary, "#FFFFFF", 0.22) if primary_set else "#1F4E8C"

    def _contrast(x, y):
        hi, lo = sorted((_lum(x), _lum(y)), reverse=True)
        return (hi + 0.05) / (lo + 0.05)

    onprimary = max(("#FFFFFF", "#0B0F14"), key=lambda c: _contrast(primary, c))

    C.update(
        ink=ink, muted=muted, line=line, grid=grid, navy=side, blue=blue,
        pos=pos, warn=warn, neg=neg, panel=panel,
        ret=_mix(ink, panel, 0.55), ret_soft=_mix(ink, panel, 0.78),
    )
    MODEL_COLORS[:] = _series_colors(primary, ink, panel)

    return (
        f"color-scheme: {'dark' if dark_panel else 'light'};"
        f"--bg:{bg};--panel:{panel};--line:{line};--line-strong:{line_strong};"
        f"--ink:{ink};--muted:{muted};--head:{ink};--side:{side};--primary:{primary};"
        f"--primary-hover:{phover};--onprimary:{onprimary};--blue:{blue};"
        f"--pos:{pos};--warn:{warn};--neg:{neg};--thead:{grid};--rowline:{grid};"
        f"--hover:{hover};--soft:{soft};--side-accent:{side_accent};"
    )


def _cfg():
    if "cfg" not in st.session_state:
        st.session_state["cfg"] = dict(CONFIG)
    return st.session_state["cfg"]


@st.cache_data(ttl=3600, show_spinner=False)
def _get_returns_cached(ticker, start, end):
    return stock_getter(ticker, start=start, end=end)


@st.cache_data(ttl=3600, show_spinner=False)
def _run_standard_model_cached(ticker, start, end, model_type, window_size, refit_freq, alpha_var, alpha_es):
    returns = _get_returns_cached(ticker, start, end)
    return run_backtest(
        returns,
        model_type=model_type,
        window_size=window_size,
        refit_freq=refit_freq,
        alpha_var=alpha_var,
        alpha_es=alpha_es,
    )


def _validate_uploaded_series(series, kind):
    if series is None:
        raise ValueError(f"{kind} : fichier vide.")
    if not isinstance(series, pd.Series):
        raise ValueError(f"{kind} : format non reconnu (une seule colonne attendue).")
    if not series.index.is_monotonic_increasing:
        series = series.sort_index()
    if series.index.has_duplicates:
        raise ValueError(f"{kind} : dates dupliquées dans l'index.")
    if series.isna().all():
        raise ValueError(f"{kind} : série entièrement NaN.")
    if not np.isfinite(series.dropna().values).all():
        raise ValueError(f"{kind} : contient des valeurs non finies.")
    return series


def parse_uploaded_file(uploaded_file, kind="series"):
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True)
    elif name.endswith(".parquet"):
        df = pd.read_parquet(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)

    series = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
    return _validate_uploaded_series(series, kind)


def _check_period(cfg):
    try:
        n = len(_get_returns_cached(cfg["ticker"], cfg["start_date"], cfg["end_date"]))
    except Exception as e:
        return False, f"Téléchargement des données impossible : {e}"
    min_obs = cfg["window_size"] + CONFIG["lags_dq"] + 20
    if n <= min_obs:
        return False, (
            f"Période trop courte : {n} observations "
            f"pour une fenêtre de {cfg['window_size']} (minimum {min_obs})."
        )
    return True, None


def backtester(selected_standards, selected_customs_dict, progress_callback=None):
    cfg = _cfg()
    tasks = []
    for code, label in STANDARD_MODELS.items():
        if code in selected_standards:
            tasks.append((code, label, None))
    for c_name, c_data in selected_customs_dict.items():
        tasks.append(("custom", c_name, c_data))

    total = len(tasks)
    results = []
    dict_outputs = {}

    for i, (m_type, display_name, c_data) in enumerate(tasks):
        pct_depart = int((i / total) * 100)
        if progress_callback:
            progress_callback(pct_depart, f"Calcul du modèle « {display_name} » en cours")

        if m_type == "custom":
            returns = _get_returns_cached(cfg["ticker"], cfg["start_date"], cfg["end_date"])
            res = run_backtest(
                returns,
                model_type="custom",
                window_size=cfg["window_size"],
                refit_freq=cfg["refit_freq"],
                alpha_var=cfg["alpha_var"],
                alpha_es=cfg["alpha_es"],
                custom_data=c_data,
            )
        else:
            res = _run_standard_model_cached(
                cfg["ticker"], cfg["start_date"], cfg["end_date"],
                m_type,
                cfg["window_size"], cfg["refit_freq"],
                cfg["alpha_var"], cfg["alpha_es"],
            )

        dict_outputs[display_name] = res
        clean_res = {
            k: v for k, v in res.items()
            if k not in ["var_series", "es_frtb_series", "test_returns", "rolling_250", "eta_hist", "lambda_hist",
                         "refit_dates"]
        }
        clean_res["model"] = display_name
        results.append(clean_res)

        pct_fin = int(((i + 1) / total) * 100)
        if progress_callback:
            progress_callback(pct_fin, f"Modèle « {display_name} » terminé")

    df_summary = pd.DataFrame(results).set_index("model").T
    return df_summary, dict_outputs


@st.cache_data(ttl=3600, show_spinner=False)
def is_valid_ticker(symbol: str) -> bool:
    if not symbol:
        return False
    try:
        hist = yf.Ticker(symbol).history(period="5d")
        return not hist.empty
    except Exception:
        return False


def _h(markup: str) -> str:
    return "".join(line.strip() for line in markup.splitlines())


def _esc(x) -> str:
    return html.escape(str(x))


def _render(markup: str):
    st.markdown(_h(markup), unsafe_allow_html=True)


def page_header(kicker, title, subtitle=""):
    sub = f'<div class="ph-sub">{_esc(subtitle)}</div>' if subtitle else ""
    _render(f"""
    <div class="ph">
        <div class="ph-kicker">{_esc(kicker)}</div>
        <div class="ph-title">{_esc(title)}</div>
        {sub}
    </div>
    """)


def section(num, title, desc=""):
    d = f'<span class="sec-desc">{_esc(desc)}</span>' if desc else ""
    _render(f"""
    <div class="sec">
        <span class="sec-num">{_esc(num)}</span>
        <span class="sec-title">{_esc(title)}</span>
        {d}
    </div>
    """)


def panel_label(text):
    _render(f'<div class="plabel">{_esc(text)}</div>')


def status_line(kind, text):
    tag = {"pos": "Valide", "neg": "Erreur", "warn": "Attention"}.get(kind, "Info")
    _render(
        f'<div class="stat {kind}"><span class="stat-tag">{tag}</span>'
        f'<span class="stat-txt">{_esc(text)}</span></div>'
    )


def context_ribbon():
    cfg = _cfg()
    fin = cfg["end_date"] or datetime.date.today()
    cells = [
        ("Instrument", cfg["ticker"]),
        ("Période", f"{cfg['start_date']} → {fin}"),
        ("Fenêtre", f"{cfg['window_size']} j"),
        ("Refit", f"{cfg['refit_freq']} j"),
        ("α VaR", f"{float(cfg['alpha_var']) * 100:.2f}%"),
        ("α ES", f"{float(cfg['alpha_es']) * 100:.2f}%"),
    ]
    inner = "".join(
        f'<div class="rb-cell"><div class="rb-k">{_esc(k)}</div>'
        f'<div class="rb-v">{_esc(v)}</div></div>'
        for k, v in cells
    )
    _render(f'<div class="ribbon">{inner}</div>')


def kpi_strip(items):
    cells = ""
    for label, value, sub, tone in items:
        cells += (
            f'<div class="kpi {tone}"><div class="kpi-k">{_esc(label)}</div>'
            f'<div class="kpi-v">{_esc(value)}</div>'
            f'<div class="kpi-s">{_esc(sub)}</div></div>'
        )
    _render(f'<div class="kpis">{cells}</div>')


def _fmt_scalar(f, key=""):
    if np.isnan(f):
        return "—"
    if "pval" in str(key):
        return f"{f:.4f}" if (f == 0 or abs(f) >= 1e-4) else f"{f:.2e}"
    if f.is_integer() and abs(f) < 1e12:
        return f"{int(f):,}".replace(",", " ")
    if f != 0 and abs(f) < 1e-4:
        return f"{f:.2e}"
    return f"{f:,.4f}".replace(",", " ")


def _fmt(key, v):
    if isinstance(v, (tuple, list, np.ndarray)):
        try:
            vals = [float(x) for x in v]
        except (TypeError, ValueError):
            return str(v)
        if all(np.isnan(x) for x in vals):
            return "—"
        return "[" + " ; ".join(_fmt_scalar(x, key) for x in vals) + "]"
    if isinstance(v, str):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return _fmt_scalar(f, key)


def _cell_tone(key, v):
    key = str(key)
    if key == "basel_zone_actu":
        return {"VERTE": "pos", "JAUNE": "warn", "ROUGE": "neg"}.get(str(v).upper(), "")
    if "pval" in key:
        try:
            return "pos" if float(v) > 0.05 else "neg"
        except (TypeError, ValueError):
            return ""
    return ""


def render_metrics_table(df_summary, compact=False):
    models = list(df_summary.columns)
    head = "".join(f'<th class="num">{_esc(m)}</th>' for m in models)
    code_th = "" if compact else "<th>Code</th>"
    rows = ""
    for key in df_summary.index:
        label = METRIC_LABELS.get(key, str(key).replace("_", " ").capitalize())
        code_td = "" if compact else f'<td class="code">{_esc(key)}</td>'
        cells = ""
        for m in models:
            raw = df_summary.loc[key, m]
            cells += f'<td class="num {_cell_tone(key, raw)}">{_esc(_fmt(key, raw))}</td>'
        rows += f'<tr><td class="lbl">{_esc(label)}</td>{code_td}{cells}</tr>'
    _render(
        f'<div class="tbl-wrap"><table class="fin"><thead><tr><th>Métrique</th>'
        f'{code_th}{head}</tr></thead><tbody>{rows}</tbody></table></div>'
    )


def render_kv_table(rows):
    body = "".join(
        f'<tr><td class="lbl">{_esc(k)}</td><td class="num">{_esc(v)}</td></tr>'
        for k, v in rows
    )
    _render(
        f'<div class="tbl-wrap"><table class="fin"><thead><tr><th>Paramètre</th>'
        f'<th class="num">Valeur</th></tr></thead><tbody>{body}</tbody></table></div>'
    )


def _plot_layout(fig, height):
    fig.update_layout(
        paper_bgcolor=C["panel"],
        plot_bgcolor=C["panel"],
        margin=dict(l=8, r=8, t=8, b=8),
        height=height,
        font=dict(family="IBM Plex Mono, monospace", size=11, color=C["muted"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        hoverlabel=dict(font=dict(family="IBM Plex Mono, monospace", size=11, color=C["ink"]),
                        bgcolor=C["panel"], bordercolor=C["line"]),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=C["line"],
                   ticks="outside", tickcolor=C["line"]),
        yaxis=dict(showgrid=True, gridcolor=C["grid"], zeroline=False,
                   linecolor=C["line"], ticks="outside", tickcolor=C["line"]),
    )
    return fig


def _neg(series):
    return -series.abs() if (series > 0).any() else series


def build_backtest_chart(raw_res, conf_label, height=470):
    if "var_series" not in raw_res or "test_returns" not in raw_res:
        return None
    returns_s = raw_res["test_returns"]
    var_plot = _neg(raw_res["var_series"])

    fig = go.Figure()
    fig.add_hline(y=0, line_color=C["line"], line_width=1)
    fig.add_trace(go.Scatter(
        x=returns_s.index, y=returns_s.values, mode="lines", name="Rendements",
        line=dict(color=C["ret"], width=0.9),
    ))
    fig.add_trace(go.Scatter(
        x=var_plot.index, y=var_plot.values, mode="lines", name=f"VaR {conf_label}",
        line=dict(color=C["navy"], width=1.6, shape="hv"),
    ))
    if "es_frtb_series" in raw_res:
        es_plot = _neg(raw_res["es_frtb_series"])
        fig.add_trace(go.Scatter(
            x=es_plot.index, y=es_plot.values, mode="lines", name="ES (FRTB)",
            line=dict(color=C["warn"], width=1.2, dash="dot", shape="hv"),
        ))
    exc_mask = returns_s < var_plot
    fig.add_trace(go.Scatter(
        x=returns_s[exc_mask].index, y=returns_s[exc_mask].values,
        mode="markers", name="Exceptions",
        marker=dict(color=C["neg"], size=7, symbol="x-thin",
                    line=dict(color=C["neg"], width=1.6)),
    ))
    return _plot_layout(fig, height)


def build_overlay_chart(dict_outputs, conf_label, height=420):
    first = next((v for v in dict_outputs.values() if "test_returns" in v), None)
    if first is None:
        return None
    fig = go.Figure()
    fig.add_hline(y=0, line_color=C["line"], line_width=1)
    r = first["test_returns"]
    fig.add_trace(go.Scatter(
        x=r.index, y=r.values, mode="lines", name="Rendements",
        line=dict(color=C["ret_soft"], width=0.8),
    ))
    for i, (name, res) in enumerate(dict_outputs.items()):
        if "var_series" not in res:
            continue
        v = _neg(res["var_series"])
        fig.add_trace(go.Scatter(
            x=v.index, y=v.values, mode="lines", name=f"VaR {conf_label} · {name}",
            line=dict(color=MODEL_COLORS[i % len(MODEL_COLORS)], width=1.5, shape="hv"),
        ))
    _plot_layout(fig, height)
    fig.update_layout(hoverlabel=dict(namelength=-1))
    return fig


def _conf_label(cfg):
    return f"{(1 - float(cfg['alpha_var'])) * 100:g}%"


def _make_progress():
    barre = st.progress(0)
    status = st.empty()

    def cb(pct, message):
        barre.progress(pct)
        status.markdown(
            f'<div class="status-line">{_esc(message)} · {pct:d}%</div>',
            unsafe_allow_html=True,
        )

    def clear():
        barre.empty()
        status.empty()

    return cb, clear


def afficher_synthese_comparative(df_summary):
    section("05", "Synthèse", "Meilleur modèle par critère d'audit")
    try:
        cfg = _cfg()
        modeles = df_summary.columns.tolist()

        kupiec = {m: float(df_summary.loc["kupiec_pval", m]) for m in modeles}
        best_kupiec = max(kupiec, key=kupiec.get)

        cc = {m: float(df_summary.loc["christof_cc_pval", m]) for m in modeles}
        best_cc = max(cc, key=cc.get)

        dq = {m: float(df_summary.loc["dq_pval", m]) for m in modeles}
        best_dq = max(dq, key=dq.get)

        exc = {m: float(df_summary.loc["exceptions", m]) for m in modeles}
        n_obs = float(df_summary.loc["obs_tested", modeles[0]])
        target = n_obs * cfg["alpha_var"]
        best_exc = min(exc, key=lambda m: abs(exc[m] - target))

        rows = [
            ("Précision de la VaR", "Kupiec POF", best_kupiec, f"p = {kupiec[best_kupiec]:.4f}"),
            ("Indépendance des exceptions", "Christoffersen CC", best_cc, f"p = {cc[best_cc]:.4f}"),
            ("Qualité dynamique", "Dynamic Quantile", best_dq, f"p = {dq[best_dq]:.4f}"),
            ("Calibration du taux d'exceptions", "Écart à la cible", best_exc,
             f"{int(exc[best_exc])} obs. · cible ≈ {target:.0f}"),
        ]
        body = "".join(
            f'<tr><td class="lbl">{_esc(a)}</td><td class="code">{_esc(b)}</td>'
            f'<td class="lbl strong">{_esc(c)}</td><td class="num">{_esc(d)}</td></tr>'
            for a, b, c, d in rows
        )
        _render(
            '<div class="tbl-wrap"><table class="fin"><thead><tr>'
            '<th>Critère</th><th>Test</th><th>Meilleur modèle</th>'
            f'<th class="num">Indicateur</th></tr></thead><tbody>{body}</tbody></table></div>'
        )
    except Exception:
        status_line("", "Synthèse automatique indisponible pour cette sélection.")


def gestionnaire_modeles_custom(show_checkboxes=True):
    if "custom_models" not in st.session_state:
        st.session_state["custom_models"] = {}

    with st.container(border=True):
        panel_label("Importer un modèle externe")
        custom_name = st.text_input("Nom du modèle", placeholder="ex. RiskMetrics_2026", key="custom_model_name")

        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            panel_label("A · Série VaR")
            file_var = st.file_uploader("Série VaR", type=["csv", "xlsx", "parquet"], key="add_var",
                                        label_visibility="collapsed")
        with c2:
            panel_label("B · Série Expected Shortfall")
            file_es = st.file_uploader("Série ES", type=["csv", "xlsx", "parquet"], key="add_es",
                                       label_visibility="collapsed")
        with c3:
            panel_label("C · Série de volatilité")
            file_vol = st.file_uploader("Série de volatilité", type=["csv", "xlsx", "parquet"], key="add_vol",
                                        label_visibility="collapsed")

        st.caption("Formats acceptés : CSV, XLSX, Parquet · index = dates, première colonne = valeurs.")

        if st.button("Enregistrer le modèle", type="primary"):
            clean_name = custom_name.strip()
            if not clean_name:
                status_line("neg", "Veuillez saisir un nom de modèle.")
            elif clean_name in st.session_state["custom_models"]:
                status_line("neg", "Un modèle portant ce nom existe déjà.")
            elif not (file_var and file_es and file_vol):
                status_line("neg", "Veuillez importer les trois séries requises.")
            else:
                try:
                    var_series = parse_uploaded_file(file_var, "VaR")
                    es_series = parse_uploaded_file(file_es, "ES")
                    vol_series = parse_uploaded_file(file_vol, "Volatilité")
                except ValueError as e:
                    status_line("neg", str(e))
                else:
                    st.session_state["custom_models"][clean_name] = {
                        "var": var_series,
                        "es": es_series,
                        "vol": vol_series,
                    }
                    st.toast(f"Modèle « {clean_name} » enregistré.")
                    st.rerun()

    if st.session_state["custom_models"]:
        panel_label("Modèles enregistrés · sélection pour l'audit" if show_checkboxes else "Modèles enregistrés")
        cols_custom_select = {}
        for c_name in list(st.session_state["custom_models"].keys()):
            with st.container(border=True):
                if show_checkboxes:
                    col_info, col_chk, col_del = st.columns([3, 2, 1], vertical_alignment="center")
                else:
                    col_info, col_del = st.columns([4, 1], vertical_alignment="center")

                with col_info:
                    _render(
                        f'<div class="card-title">{_esc(c_name)}</div>'
                        '<div class="card-desc">Modèle externe · VaR / ES / Vol</div>'
                    )

                if show_checkboxes:
                    with col_chk:
                        cols_custom_select[c_name] = st.checkbox("Inclure dans l'audit", value=True,
                                                                 key=f"chk_{c_name}")

                with col_del:
                    if st.button("Retirer", key=f"del_{c_name}", use_container_width=True):
                        del st.session_state["custom_models"][c_name]
                        st.rerun()
        return cols_custom_select
    return {}


def page_audit_comparatif():
    cfg = _cfg()
    page_header("Validation de modèles", "Audit comparatif",
                "Backtest simultané de plusieurs modèles de VaR / ES sur un même instrument.")
    context_ribbon()

    section("01", "Modèles de référence", "Méthodologies à comparer")
    defs = [
        ("historical", "Simulation historique", "Quantiles empiriques"),
        ("garch_t", "GARCH-t", "Volatilité conditionnelle · Student-t"),
        ("gjr_skewt", "GJR-GARCH", "Volatilité asymétrique · Skew-t"),
    ]
    selected_standards = []
    for col, (code, title, desc) in zip(st.columns(3, gap="small"), defs):
        with col:
            with st.container(border=True):
                _render(
                    f'<div class="card-title">{_esc(title)}</div>'
                    f'<div class="card-desc">{_esc(desc)}</div>'
                )
                if st.checkbox("Inclure dans l'audit", value=True, key=f"model_{code}"):
                    selected_standards.append(code)

    section("02", "Modèles custom", "Séries de risque externes")
    custom_selections = gestionnaire_modeles_custom()
    selected_customs = {
        name: st.session_state["custom_models"][name]
        for name, is_sel in custom_selections.items() if is_sel
    }

    section("03", "Exécution")
    if st.button("Lancer l'audit comparatif", type="primary"):
        if not selected_standards and not selected_customs:
            status_line("warn", "Veuillez sélectionner au moins un modèle à auditer.")
        else:
            ok, msg = _check_period(cfg)
            if not ok:
                status_line("neg", msg)
            else:
                cb, clear = _make_progress()
                try:
                    df_summary, dict_outputs = backtester(selected_standards, selected_customs, progress_callback=cb)
                except ValueError as e:
                    clear()
                    status_line("neg", str(e))
                else:
                    clear()
                    st.session_state["last_comp"] = {
                        "df": df_summary,
                        "out": dict_outputs,
                        "conf": _conf_label(cfg),
                        "ticker": cfg["ticker"],
                    }

    res = st.session_state.get("last_comp")
    if res:
        section("04", "Résultats", f"{res['ticker']} · tableau comparatif des métriques")
        render_metrics_table(res["df"])
        st.caption("Vert : H₀ non rejetée (p > 5 %) · Rouge : H₀ rejetée (p ≤ 5 %).")
        afficher_synthese_comparative(res["df"])
        fig = build_overlay_chart(res["out"], res["conf"])
        if fig is not None:
            section("06", "VaR comparées", f"Seuils VaR {res['conf']} superposés aux rendements")
            st.plotly_chart(fig, use_container_width=True, theme=None)


def page_audit_unitaire():
    cfg = _cfg()
    page_header("Validation de modèles", "Audit unitaire",
                "Backtest détaillé d'un modèle : exceptions, tests statistiques, zone de Bâle.")
    context_ribbon()

    section("01", "Sélection du modèle")
    options_map = {label: code for code, label in STANDARD_MODELS.items()}
    custom_models = st.session_state.get("custom_models", {})
    for c_name in custom_models.keys():
        options_map[c_name] = "custom"

    modele_choisi_label = st.radio(
        "Sélectionnez le modèle :",
        options=list(options_map.keys()),
        horizontal=True,
        label_visibility="collapsed",
    )
    modele_code = options_map[modele_choisi_label]

    section("02", "Modèles custom", "Séries de risque externes")
    gestionnaire_modeles_custom(show_checkboxes=False)

    section("03", "Exécution")
    if st.button("Lancer l'audit unitaire", type="primary"):
        ok, msg = _check_period(cfg)
        if not ok:
            status_line("neg", msg)
        else:
            cb, clear = _make_progress()
            selected_stds = [modele_code] if modele_code != "custom" else []
            selected_cust = (
                {modele_choisi_label: custom_models[modele_choisi_label]} if modele_code == "custom" else {})
            try:
                df_summary, dict_outputs = backtester(selected_stds, selected_cust, progress_callback=cb)
            except ValueError as e:
                clear()
                status_line("neg", str(e))
            else:
                clear()
                st.session_state["last_unit"] = {
                    "df": df_summary,
                    "out": dict_outputs,
                    "label": modele_choisi_label,
                    "conf": _conf_label(cfg),
                    "ticker": cfg["ticker"],
                }

    res = st.session_state.get("last_unit")
    if not res:
        return

    label = res["label"]
    df_summary = res["df"]
    section("04", "Résultats", f"{label} · {res['ticker']}")

    zone = str(df_summary.loc["basel_zone_actu", label])
    n_exc = int(float(df_summary.loc["exceptions", label]))
    rate = float(df_summary.loc["exc_rate", label])
    kupiec_p = float(df_summary.loc["kupiec_pval", label])
    try:
        n_obs = float(df_summary.loc["obs_tested", label])
        target_txt = f"cible ≈ {n_obs * cfg['alpha_var']:.0f}"
    except Exception:
        target_txt = "—"

    kpi_strip([
        ("Zone de Bâle", zone, "Feu réglementaire actuel",
         {"VERTE": "pos", "JAUNE": "warn", "ROUGE": "neg"}.get(zone.upper(), "")),
        ("Exceptions VaR", f"{n_exc}", target_txt, ""),
        ("Taux d'exceptions", f"{rate:.2f}%", f"VaR {res['conf']}", ""),
        ("Kupiec POF · p-value", f"{kupiec_p:.4f}", "H₀ non rejetée" if kupiec_p > 0.05 else "H₀ rejetée",
         "pos" if kupiec_p > 0.05 else "neg"),
    ])

    panel_label(f"Backtest VaR {res['conf']} & Expected Shortfall")
    fig = build_backtest_chart(res["out"].get(label, {}), res["conf"], height=440)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, theme=None)
    else:
        status_line("", "Séries VaR / rendements indisponibles pour ce modèle.")

    section("05", "Décomposition des métriques", "Toutes les statistiques du backtest")
    half = (len(df_summary.index) + 1) // 2
    col_l, col_r = st.columns(2, gap="medium")
    with col_l:
        render_metrics_table(df_summary.iloc[:half], compact=True)
    with col_r:
        render_metrics_table(df_summary.iloc[half:], compact=True)


def page_config():
    page_header("Paramétrage", "Configuration", "Instrument, période d'analyse et paramètres de backtest.")
    cfg = _cfg()

    col_a, col_b = st.columns(2, gap="medium")

    with col_a:
        section("01", "Période d'analyse")
        raw_start = st.text_input("Date de début (AAAA-MM-JJ)", str(cfg["start_date"]))
        raw_end = st.text_input("Date de fin (AAAA-MM-JJ, ou None)", str(cfg["end_date"]))

        try:
            new_start = datetime.date.fromisoformat(raw_start.strip())
            new_end = (None if raw_end.strip().lower() == "none" else datetime.date.fromisoformat(raw_end.strip()))
        except ValueError:
            status_line("neg", "Format invalide, utilisez AAAA-MM-JJ. Les dates précédentes sont conservées.")
        else:
            end_eff = new_end or datetime.date.today()
            jours_bourse = (end_eff - new_start).days * 252 / 365
            if end_eff <= new_start:
                status_line("neg",
                            "La date de fin doit être postérieure à la date de début. Les dates précédentes sont conservées.")
            elif jours_bourse <= cfg["window_size"]:
                status_line("neg",
                            f"Période trop courte : ~{int(jours_bourse)} jours de bourse pour une fenêtre de {cfg['window_size']}. Les dates précédentes sont conservées.")
            else:
                cfg["start_date"] = new_start
                cfg["end_date"] = new_end
                status_line("pos", "Période valide.")

    with col_b:
        section("02", "Instrument")
        raw = st.text_input("Ticker (Yahoo Finance)", cfg["ticker"])
        symbol = raw.strip().upper()
        if not symbol:
            status_line("warn", "Veuillez saisir un ticker.")
        elif is_valid_ticker(symbol):
            cfg["ticker"] = symbol
            status_line("pos", f"Ticker valide : {symbol}")
        else:
            status_line("neg", f"Ticker introuvable : « {symbol} ». Le ticker précédent est conservé.")

    section("03", "Paramètres de backtest", "Lecture seule · définis dans main.CONFIG")
    render_kv_table([
        ("Fenêtre d'estimation (jours)", cfg["window_size"]),
        ("Fréquence de réestimation (jours)", cfg["refit_freq"]),
        ("Niveau α VaR", cfg["alpha_var"]),
        ("Niveau α ES", cfg["alpha_es"]),
        ("Retards du test DQ", CONFIG.get("lags_dq", "—")),
    ])


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    /*ROOT*/
    --mono: 'IBM Plex Mono', ui-monospace, Consolas, monospace;
    --sans: 'IBM Plex Sans', 'Segoe UI', Arial, sans-serif;
}

html, body, [class*="css"], .stApp { font-family: var(--sans); }
.stApp { background: var(--bg); color: var(--ink); }

[data-testid="stMain"] p, [data-testid="stMain"] label, [data-testid="stMain"] li,
[data-testid="stMain"] span, section.main p, section.main label, section.main span { color: var(--ink); }
[data-testid="stMain"] [data-testid="stCaptionContainer"] *, section.main [data-testid="stCaptionContainer"] * { color: var(--muted) !important; }
[data-testid="stFileUploader"] * { color: var(--ink); }
[data-testid="stFileUploader"] small, [data-testid="stFileUploaderFile"] small,
[data-testid="stFileUploaderDropzoneInstructions"] * { color: var(--muted) !important; }
[data-testid="stFileUploader"] svg { fill: var(--muted); }
[data-testid="stAlert"] { background: var(--panel) !important; }
[data-testid="stAlert"] * { color: var(--ink) !important; }
.stTextInput input::placeholder { color: var(--muted) !important; }
[data-testid="stToast"], [data-testid="stToast"] * { color: var(--ink) !important; }

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
[data-testid="stHeader"] { background: transparent; height: 2rem; }

.block-container { max-width: 1480px; padding: 1.4rem 2.2rem 4rem 2.2rem; }

[data-testid="stSidebar"] { background: var(--side); border-right: none; }
[data-testid="stSidebar"] * { color: #C4CFDD; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; margin: 1.2rem 0; }
[data-testid="stSidebar"] .sb-brand { color: #FFFFFF; font-size: 17px; font-weight: 600; letter-spacing: 0.14em; }
[data-testid="stSidebar"] .sb-sub { color: #8DA2BC; font-size: 10.5px; letter-spacing: 0.06em; margin-top: 4px; line-height: 1.5; }
[data-testid="stSidebar"] .sb-author { color: #6F85A3; font-size: 10px; letter-spacing: 0.08em; margin-top: 14px; text-transform: uppercase; }
[data-testid="stSidebar"] .sb-label { color: #6F85A3; font-size: 10px; letter-spacing: 0.16em; text-transform: uppercase; margin-bottom: 8px; }
[data-testid="stSidebar"] .sb-tick { color: #FFFFFF; font-family: var(--mono); font-size: 20px; font-weight: 600; letter-spacing: 0.02em; }
[data-testid="stSidebar"] .sb-per { color: #8DA2BC; font-family: var(--mono); font-size: 11px; margin-top: 4px; }
[data-testid="stSidebar"] .sb-foot { color: #587190; font-size: 10px; line-height: 1.6; }

[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 9px 12px; border-radius: 0; border: none;
    border-left: 2px solid transparent; background: transparent; width: 100%;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(255,255,255,0.05); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: rgba(255,255,255,0.09); border-left: 2px solid var(--side-accent);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) * { color: #FFFFFF; font-weight: 500; }
[data-testid="stSidebar"] [role="radiogroup"] label p { font-size: 13px; letter-spacing: 0.02em; }

[role="radiogroup"] label > div:first-child { display: none; }
[data-testid="stMain"] [role="radiogroup"],
section.main [role="radiogroup"] { gap: 0; flex-wrap: wrap; }
[data-testid="stMain"] [role="radiogroup"] label,
section.main [role="radiogroup"] label {
    background: var(--panel); border: 1px solid var(--line); border-radius: 0;
    padding: 8px 18px; margin: 0 -1px 0 0;
}
[data-testid="stMain"] [role="radiogroup"] label:has(input:checked),
section.main [role="radiogroup"] label:has(input:checked) { background: var(--primary); border-color: var(--primary); }
[data-testid="stMain"] [role="radiogroup"] label:has(input:checked) p,
section.main [role="radiogroup"] label:has(input:checked) p { color: var(--onprimary); }
[data-testid="stMain"] [role="radiogroup"] label p,
section.main [role="radiogroup"] label p { font-size: 12.5px; font-weight: 500; letter-spacing: 0.02em; color: var(--ink); }
[data-testid="stMain"] [role="radiogroup"] label:has(input:checked) p,
[data-testid="stMain"] [role="radiogroup"] label:has(input:checked) span,
section.main [role="radiogroup"] label:has(input:checked) span { color: var(--onprimary) !important; }

.ph { padding: 4px 0 18px 0; border-bottom: 2px solid var(--head); margin-bottom: 0; }
.ph-kicker { font-size: 10.5px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--blue); font-weight: 600; }
.ph-title { font-size: 26px; font-weight: 600; color: var(--head); letter-spacing: -0.01em; margin-top: 4px; }
.ph-sub { font-size: 13px; color: var(--muted); margin-top: 6px; }

.ribbon { display: flex; flex-wrap: wrap; background: var(--panel); border: 1px solid var(--line); border-top: none; margin-bottom: 6px; }
.rb-cell { flex: 1 1 130px; padding: 9px 16px; border-right: 1px solid var(--line); }
.rb-cell:last-child { border-right: none; }
.rb-k { font-size: 9.5px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); }
.rb-v { font-family: var(--mono); font-size: 13px; font-weight: 500; color: var(--ink); margin-top: 2px; white-space: nowrap; }

.sec { display: flex; align-items: baseline; gap: 12px; margin: 30px 0 12px 0; padding-bottom: 7px; border-bottom: 1px solid var(--line-strong); }
.sec-num { font-family: var(--mono); font-size: 11px; color: var(--blue); font-weight: 600; }
.sec-title { font-size: 12px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: var(--head); }
.sec-desc { font-size: 12px; color: var(--muted); margin-left: auto; }
.plabel { font-size: 10.5px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); font-weight: 600; margin: 2px 0 8px 0; }

[data-testid="stVerticalBlockBorderWrapper"] { background: var(--panel); border: 1px solid var(--line) !important; border-radius: 0 !important; }
.card-title { font-size: 13px; font-weight: 600; color: var(--head); letter-spacing: 0.02em; }
.card-desc { font-size: 11.5px; color: var(--muted); margin: 2px 0 8px 0; }

.kpis { display: flex; flex-wrap: wrap; background: var(--panel); border: 1px solid var(--line); margin-bottom: 18px; }
.kpi { flex: 1 1 200px; padding: 14px 18px; border-right: 1px solid var(--line); border-top: 3px solid var(--line-strong); }
.kpi:last-child { border-right: none; }
.kpi.pos { border-top-color: var(--pos); }
.kpi.warn { border-top-color: var(--warn); }
.kpi.neg { border-top-color: var(--neg); }
.kpi-k { font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
.kpi-v { font-family: var(--mono); font-size: 26px; font-weight: 600; color: var(--ink); margin-top: 6px; letter-spacing: -0.01em; }
.kpi.pos .kpi-v { color: var(--pos); }
.kpi.warn .kpi-v { color: var(--warn); }
.kpi.neg .kpi-v { color: var(--neg); }
.kpi-s { font-size: 11px; color: var(--muted); margin-top: 2px; }

.tbl-wrap { background: var(--panel); border: 1px solid var(--line); overflow-x: auto; }
table.fin { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
table.fin thead th {
    background: var(--thead); color: var(--head); text-align: left; font-size: 10px; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase; padding: 9px 14px; border-bottom: 2px solid var(--head); white-space: nowrap;
}
table.fin td { padding: 8px 14px; border-bottom: 1px solid var(--rowline); font-size: 12.5px; color: var(--ink); }
table.fin tbody tr:last-child td { border-bottom: none; }
table.fin tbody tr:hover td { background: var(--hover); }
table.fin .num { text-align: right; font-family: var(--mono); font-size: 12.5px; white-space: nowrap; }
table.fin .code { font-family: var(--mono); font-size: 10.5px; color: var(--muted); }
table.fin .lbl { font-family: var(--sans); font-size: 12.5px; }
table.fin .strong { font-weight: 600; color: var(--head); }
table.fin .pos { color: var(--pos); font-weight: 600; }
table.fin .warn { color: var(--warn); font-weight: 600; }
table.fin .neg { color: var(--neg); font-weight: 600; }

[data-testid="stWidgetLabel"] p, .stTextInput label p, .stCheckbox label p { font-size: 12px; color: var(--ink); }
[data-testid="stWidgetLabel"] p { font-weight: 500; letter-spacing: 0.02em; }

.stTextInput input, .stNumberInput input, .stDateInput input, .stTextArea textarea {
    background: var(--panel) !important; color: var(--ink) !important; border: 1px solid var(--line-strong) !important;
    border-radius: 0 !important; font-family: var(--mono); font-size: 13px;
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
    border-color: var(--blue) !important; box-shadow: 0 0 0 1px var(--blue) !important;
}
[data-baseweb="input"], [data-baseweb="base-input"] { border-radius: 0 !important; background: var(--panel) !important; }
[data-baseweb="input"] { border: 1px solid var(--line-strong) !important; box-shadow: none !important; }
[data-baseweb="input"]:focus-within { border-color: var(--blue) !important; box-shadow: 0 0 0 1px var(--blue) !important; }
.stTextInput input, .stTextInput input:focus { outline: none !important; border: none !important; box-shadow: none !important; }

.stat { display: flex; align-items: baseline; gap: 12px; padding: 9px 0; border-bottom: 1px solid var(--line); margin-bottom: 4px; }
.stat-tag { font-family: var(--mono); font-size: 9.5px; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); border: 1px solid currentColor; padding: 2px 7px; min-width: 66px; text-align: center; }
.stat.pos .stat-tag { color: var(--pos); }
.stat.neg .stat-tag { color: var(--neg); }
.stat.warn .stat-tag { color: var(--warn); }
.stat-txt { font-size: 12.5px; color: var(--ink); }

.stButton button, [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {
    border-radius: 0 !important; font-size: 11.5px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; padding: 0.55rem 1.4rem; transition: background 0.12s ease, border-color 0.12s ease;
}
.stButton button[kind="secondary"], [data-testid="stBaseButton-secondary"] { background: var(--panel); border: 1px solid var(--line-strong); color: var(--head); }
.stButton button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover { border-color: var(--head); background: var(--hover); color: var(--head); }
.stButton button[kind="primary"], [data-testid="stBaseButton-primary"] { background: var(--primary); border: 1px solid var(--primary); color: var(--onprimary); }
.stButton button[kind="primary"] p, [data-testid="stBaseButton-primary"] p { color: var(--onprimary); }
.stButton button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover { background: var(--primary-hover); border-color: var(--primary-hover); color: var(--onprimary); }

[data-testid="stCheckbox"] label p { font-size: 12.5px; }
[data-testid="stFileUploader"] section { background: var(--soft); border: 1px dashed var(--line-strong); border-radius: 0; padding: 0.55rem; }
[data-testid="stFileUploader"] section:hover { border-color: var(--blue); }
[data-testid="stFileUploader"] button { border-radius: 0; }
[data-testid="stFileUploaderDropzoneInstructions"] span, [data-testid="stFileUploaderDropzoneInstructions"] small { font-size: 11px; }

[data-testid="stAlert"] { border-radius: 0; border: 1px solid var(--line); border-left-width: 3px; font-size: 13px; }
[data-testid="stProgress"] > div > div { background: var(--thead); border-radius: 0; }
[data-testid="stProgress"] > div > div > div { background: var(--blue); border-radius: 0; }
.status-line { font-family: var(--mono); font-size: 11.5px; color: var(--muted); margin: 4px 0 10px 0; }

[data-testid="stCaptionContainer"] { color: var(--muted) !important; font-size: 11.5px; }
[data-testid="stPlotlyChart"] { background: var(--panel); border: 1px solid var(--line); padding: 8px; }
hr { border-color: var(--line) !important; }

@media (max-width: 768px) {
    .block-container { padding: 1rem 1rem 3rem 1rem; }
    .ph-title { font-size: 21px; }
    .sec-desc { display: none; }
}
</style>
"""


def afficher_interface():
    st.set_page_config(page_title="Audit Operator", layout="wide")

    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "dark"

    st.markdown(CSS.replace("/*ROOT*/", apply_theme()), unsafe_allow_html=True)

    cfg = _cfg()

    st.sidebar.markdown(_h("""
    <div>
        <div class="sb-brand">AUDIT OPERATOR</div>
        <div class="sb-sub">Validation des modèles<br>de VaR et d'Expected Shortfall</div>
        <div class="sb-author">Adam Khazani Ahsaini</div>
    </div>
    """), unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sb-label">Navigation</div>', unsafe_allow_html=True)

    menu = st.sidebar.radio(
        "Navigation",
        ["Audit Comparatif", "Audit Unitaire", "Config"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")

    is_dark = st.sidebar.toggle("Thème sombre", value=(st.session_state["theme_mode"] == "dark"))
    if is_dark and st.session_state["theme_mode"] != "dark":
        st.session_state["theme_mode"] = "dark"
        st.rerun()
    elif not is_dark and st.session_state["theme_mode"] != "light":
        st.session_state["theme_mode"] = "light"
        st.rerun()

    st.sidebar.markdown("---")

    date_fin = cfg["end_date"] or datetime.date.today()
    st.sidebar.markdown(_h(f"""
    <div class="sb-label">Marché actif</div>
    <div class="sb-tick">{_esc(cfg['ticker'])}</div>
    <div class="sb-per">{_esc(cfg['start_date'])} → {_esc(date_fin)}</div>
    """), unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div class="sb-foot">Données : Yahoo Finance<br>'
        'VaR : Kupiec · Christoffersen · DQ<br>'
        'ES : McNeil-Frey · Acerbi-Szekely<br>'
        'Cadre : Bâle / FRTB</div>',
        unsafe_allow_html=True,
    )

    if menu == "Audit Comparatif":
        page_audit_comparatif()
    elif menu == "Audit Unitaire":
        page_audit_unitaire()
    elif menu == "Config":
        page_config()


if __name__ == "__main__":
    if not st.runtime.exists():
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
    else:
        afficher_interface()