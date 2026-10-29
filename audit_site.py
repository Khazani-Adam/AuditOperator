import datetime
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from main import CONFIG, run_backtest, stock_getter
from streamlit.web import cli as stcli
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf

STANDARD_MODELS = {
    "historical": "Historique (HS)",
    "garch_t": "GARCH-t",
    "gjr_skewt": "GJR-GARCH (Skew-t)",
}


def parse_uploaded_file(uploaded_file):
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

    if isinstance(df, pd.DataFrame):
        return df.iloc[:, 0]
    return df


def backtester(
    selected_standards, selected_customs_dict, progress_callback=None
):
    returns = stock_getter(
        CONFIG["ticker"], start=CONFIG["start_date"], end=CONFIG["end_date"]
    )
    results = []
    dict_outputs = {}

    tasks = []
    for code, label in STANDARD_MODELS.items():
        if code in selected_standards:
            tasks.append((code, label, None))

    for c_name, c_data in selected_customs_dict.items():
        tasks.append(("custom", c_name, c_data))

    total = len(tasks)

    for i, (m_type, display_name, c_data) in enumerate(tasks):
        pct_depart = int((i / total) * 100)
        if progress_callback:
            progress_callback(
                pct_depart, f"Calcul du modèle '{display_name}' en cours..."
            )

        res = run_backtest(
            returns,
            model_type=m_type,
            window_size=CONFIG["window_size"],
            refit_freq=CONFIG["refit_freq"],
            alpha_var=CONFIG["alpha_var"],
            alpha_es=CONFIG["alpha_es"],
            custom_data=c_data,
        )
        dict_outputs[display_name] = res

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
        clean_res["model"] = display_name
        results.append(clean_res)

        pct_fin = int(((i + 1) / total) * 100)
        if progress_callback:
            progress_callback(pct_fin, f"Modèle '{display_name}' terminé.")

    df_summary = pd.DataFrame(results).set_index("model").T
    return df_summary, dict_outputs


def afficher_synthese_comparative(df_summary):
    st.markdown("---")
    st.subheader("Synthèse : Meilleur modèle par critère d'audit")

    try:
        modeles_display = df_summary.columns.tolist()

        kupiec_pvals = {
            m: float(df_summary.loc["kupiec_pval", m]) for m in modeles_display
        }
        best_kupiec = max(kupiec_pvals, key=kupiec_pvals.get)

        cc_pvals = {
            m: float(df_summary.loc["christof_cc_pval", m])
            for m in modeles_display
        }
        best_cc = max(cc_pvals, key=cc_pvals.get)

        dq_pvals = {
            m: float(df_summary.loc["dq_pval", m]) for m in modeles_display
        }
        best_dq = max(dq_pvals, key=dq_pvals.get)

        exceptions = {
            m: float(df_summary.loc["exceptions", m]) for m in modeles_display
        }
        best_exc = min(exceptions, key=lambda x: abs(exceptions[x] - 50))

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Précision VaR (Kupiec)",
                value=best_kupiec,
                help=f"Meilleure p-value : {kupiec_pvals[best_kupiec]:.4f}",
            )

        with col2:
            st.metric(
                label="Indépendance (Christoffersen)",
                value=best_cc,
                help=f"Meilleure p-value CC : {cc_pvals[best_cc]:.4f}",
            )

        with col3:
            st.metric(
                label="Dynamic Quantile Test",
                value=best_dq,
                help=f"Meilleure p-value DQ : {dq_pvals[best_dq]:.4f}",
            )

        with col4:
            st.metric(
                label="Calibration Taux d'Exceptions",
                value=best_exc,
                help=f"Exceptions observées : {int(exceptions[best_exc])}",
            )

    except Exception:
        st.info("Synthèse automatique indisponible pour cette sélection.")



def gestionnaire_modeles_custom():
    if "custom_models" not in st.session_state:
        st.session_state["custom_models"] = {}

    # Panneau d'import
    with st.container(border=True):
        st.markdown("### Custom Model")
        st.caption(
            "Import external risk series for comparison with reference models."
        )

        custom_name = st.text_input(
            "MODEL NAME",
            placeholder="e.g. RiskMetrics_2026",
            key="custom_model_name",
        )

        st.markdown("**INPUT SERIES**")

        c1, c2, c3 = st.columns(3, gap="small")

        with c1:
            st.markdown("**01 / VaR**")
            file_var = st.file_uploader(
                "Import VaR series",
                type=["csv", "xlsx", "parquet"],
                key="add_var",
            )

        with c2:
            st.markdown("**02 / Expected Shortfall**")
            file_es = st.file_uploader(
                "Import ES series",
                type=["csv", "xlsx", "parquet"],
                key="add_es",
            )

        with c3:
            st.markdown("**03 / Volatility**")
            file_vol = st.file_uploader(
                "Import volatility series",
                type=["csv", "xlsx", "parquet"],
                key="add_vol",
            )

        st.caption("Supported formats: CSV, XLSX, Parquet")

        if st.button(
            "Save custom model",
            type="primary",
            use_container_width=False,
        ):
            clean_name = custom_name.strip()

            if not clean_name:
                st.error("Please enter a model name.")
            elif clean_name in st.session_state["custom_models"]:
                st.error("A model with this name already exists.")
            elif not (file_var and file_es and file_vol):
                st.error("Please import all three required series.")
            else:
                st.session_state["custom_models"][clean_name] = {
                    "var": parse_uploaded_file(file_var),
                    "es": parse_uploaded_file(file_es),
                    "vol": parse_uploaded_file(file_vol),
                }
                st.success(f"Model '{clean_name}' saved.")
                st.rerun()

    # Registered models
    if st.session_state["custom_models"]:
        st.markdown("### Registered Custom Models")
        st.caption("Select the models to include in the comparative audit.")

        cols_custom_select = {}

        for c_name in list(st.session_state["custom_models"].keys()):
            with st.container(border=True):
                col_info, col_chk, col_del = st.columns(
                    [3, 2, 1],
                    vertical_alignment="center",
                )

                with col_info:
                    st.markdown(f"**{c_name}**")
                    st.caption("Custom risk model")

                with col_chk:
                    is_selected = st.checkbox(
                        "Include in audit",
                        value=True,
                        key=f"chk_{c_name}",
                    )
                    cols_custom_select[c_name] = is_selected

                with col_del:
                    if st.button(
                        "Remove",
                        key=f"del_{c_name}",
                        use_container_width=True,
                    ):
                        del st.session_state["custom_models"][c_name]
                        st.rerun()

        return cols_custom_select

    return {}



def page_audit_comparatif():
    st.title("Risk Model Comparison")
    st.caption(
        f"MODEL VALIDATION  /  {CONFIG['ticker']}  /  "
        f"{CONFIG['start_date']} — {CONFIG['end_date'] or 'Aujourd’hui'}"
    )


    st.markdown("### Modèles de référence")
    st.caption("Sélectionnez les méthodologies à comparer.")

    col_chk1, col_chk2, col_chk3 = st.columns(3, gap="small")

    with col_chk1:
        with st.container(border=True):
            st.markdown("**HISTORICAL SIMULATION**")
            st.caption("Quantiles empiriques")
            c_hist = st.checkbox(
                "Inclure à l'audit",
                value=True,
                key="model_historical"
            )

    with col_chk2:
        with st.container(border=True):
            st.markdown("**GARCH-t**")
            st.caption("Volatilité conditionnelle")
            c_gt = st.checkbox(
                "Inclure à l'audit",
                value=True,
                key="model_garch_t"
            )

    with col_chk3:
        with st.container(border=True):
            st.markdown("**GJR-GARCH**")
            st.caption("Volatilité asymétrique · Skew-t")
            c_gjr = st.checkbox(
                "Inclure à l'audit",
                value=True,
                key="model_gjr_skewt"
            )


    selected_standards = []
    if c_hist:
        selected_standards.append("historical")
    if c_gt:
        selected_standards.append("garch_t")
    if c_gjr:
        selected_standards.append("gjr_skewt")

    st.subheader("2. Modèles Custom")
    custom_selections = gestionnaire_modeles_custom()

    selected_customs = {
        name: st.session_state["custom_models"][name]
        for name, is_sel in custom_selections.items()
        if is_sel
    }

    st.markdown("---")

    if st.button("Lancer l'audit comparatif", type="primary"):
        if not selected_standards and not selected_customs:
            st.warning("Veuillez sélectionner au moins un modèle à auditer.")
        else:
            barre = st.progress(0)
            status = st.empty()

            def mettre_a_jour_ui(pct, message):
                barre.progress(pct)
                status.text(f"{message} ({pct}%)")

            df_summary, dict_outputs = backtester(
                selected_standards,
                selected_customs,
                progress_callback=mettre_a_jour_ui,
            )

            status.empty()
            st.success("Audit comparatif terminé.")

            st.subheader("Tableau comparatif des métriques")
            df_display = df_summary.copy()
            df_display.index.name = "Métrique"
            df_display = df_display.reset_index()
            df_display = df_display.astype(str)  # 👈 LIGNE AJOUTÉE

            st.dataframe(
                df_display,
                width="stretch",
                hide_index=True,
                height=420,
            )
            afficher_synthese_comparative(df_summary)
@st.cache_data(ttl=3600, show_spinner=False)
def is_valid_ticker(symbol: str) -> bool:
    if not symbol:
        return False
    try:
        hist = yf.Ticker(symbol).history(period="5d")
        return not hist.empty
    except Exception:
        return False

def page_config():
    st.title("Configuration")

    raw_start = st.text_input("Date de début de l'analyse (AAAA-MM-JJ)", str(CONFIG["start_date"]))
    raw_end = st.text_input("Date de fin de l'analyse (AAAA-MM-JJ, ou None)", str(CONFIG["end_date"]))

    try:
        new_start = datetime.date.fromisoformat(raw_start.strip())
        if raw_end.strip().lower() == "none":
            new_end = None
        else:
            new_end = datetime.date.fromisoformat(raw_end.strip())
    except ValueError:
        st.error("Format invalide, utilisez AAAA-MM-JJ. Les dates précédentes sont conservées.")
    else:
        end_eff = new_end or datetime.date.today()
        jours_bourse = (end_eff - new_start).days * 252 / 365

        if end_eff <= new_start:
            st.error("La date de fin doit être postérieure à la date de début. Les dates précédentes sont conservées.")
        elif jours_bourse <= CONFIG["window_size"]:
            st.error(
                f"Période trop courte : ~{int(jours_bourse)} jours de bourse pour une "
                f"fenêtre de {CONFIG['window_size']}. Les dates précédentes sont conservées."
            )
        else:
            CONFIG["start_date"] = new_start
            CONFIG["end_date"] = new_end
            st.success("Période valide.")
    raw = st.text_input("Stock à utiliser", CONFIG["ticker"])
    symbol = raw.strip().upper()

    if not symbol:
        st.warning("Veuillez saisir un ticker.")
    elif is_valid_ticker(symbol):
        CONFIG["ticker"] = symbol
        st.success(f"Ticker valide : {symbol}")
    else:
        st.error(f"Ticker introuvable : '{symbol}'. Le ticker précédent est conservé.")

def page_audit_unitaire():
    st.title("Audit Unitaire d'un Modèle")

    options_map = {label: code for code, label in STANDARD_MODELS.items()}

    custom_models = st.session_state.get("custom_models", {})
    for c_name in custom_models.keys():
        options_map[c_name] = "custom"

    modele_choisi_label = st.radio(
        "Sélectionnez le modèle à auditer :",
        options=list(options_map.keys()),
        horizontal=True,
    )

    modele_code = options_map[modele_choisi_label]

    if st.button("Lancer l'audit unitaire", type="primary"):
        barre = st.progress(0)
        status = st.empty()

        def mettre_a_jour_ui(pct, message):
            barre.progress(pct)
            status.text(f"{message} ({pct}%)")

        selected_stds = [modele_code] if modele_code != "custom" else []
        selected_cust = (
            {modele_choisi_label: custom_models[modele_choisi_label]}
            if modele_code == "custom"
            else {}
        )

        df_summary, dict_outputs = backtester(
            selected_stds, selected_cust, progress_callback=mettre_a_jour_ui
        )

        status.empty()
        st.success(f"Audit de **{modele_choisi_label}** terminé.")

        st.subheader("Performance & Conformité Bâloise")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        zone_bale = str(df_summary.loc["basel_zone_actu", modele_choisi_label])
        nb_exceptions = str(df_summary.loc["exceptions", modele_choisi_label])
        rate_exc = (
            f"{float(df_summary.loc['exc_rate', modele_choisi_label]):.2f}%"
        )
        kupiec_p = (
            f"{float(df_summary.loc['kupiec_pval', modele_choisi_label]):.4f}"
        )

        with kpi1:
            st.metric("Zone réglementaire Bâle", zone_bale)
        with kpi2:
            st.metric("Nombre d'exceptions VaR", nb_exceptions)
        with kpi3:
            st.metric("Taux d'exception", rate_exc)
        with kpi4:
            st.metric("p-value Kupiec (VaR)", kupiec_p)

        st.markdown("---")

        st.subheader("Détail complet des résultats d'audit")
        df_display = df_summary.copy()
        df_display.index.name = "Métrique"
        df_display = df_display.reset_index()
        df_display = df_display.astype(str)  # 👈 LIGNE AJOUTÉE

        st.dataframe(
            df_display,
            width="stretch",
            hide_index=True,
            height=420,
        )


        raw_res = dict_outputs.get(modele_choisi_label, {})
        if "var_series" in raw_res and "test_returns" in raw_res:
            st.subheader(
                f"Backtest VaR 99% & Expected Shortfall - {modele_choisi_label}"
            )

            returns_s = raw_res["test_returns"]
            var_s = raw_res["var_series"]
            var_plot = -var_s.abs() if (var_s > 0).any() else var_s

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(
                returns_s.index,
                returns_s.values,
                color="gray",
                alpha=0.4,
                lw=0.7,
                label="Rendements Réels",
            )
            ax.plot(
                var_plot.index,
                var_plot.values,
                color="darkred",
                lw=1.2,
                label="VaR 99%",
            )

            if "es_frtb_series" in raw_res:
                es_s = raw_res["es_frtb_series"]
                es_plot = -es_s.abs() if (es_s > 0).any() else es_s
                ax.plot(
                    es_plot.index,
                    es_plot.values,
                    color="orange",
                    lw=1.2,
                    linestyle="--",
                    label="ES 97.5% (FRTB)",
                )

            exc_mask = returns_s < var_plot
            ax.scatter(
                returns_s[exc_mask].index,
                returns_s[exc_mask].values,
                color="darkred",
                s=20,
                zorder=5,
                label="Exceptions VaR 99%",
            )

            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="lower left")
            ax.set_ylabel("Rendements / VaR")

            st.pyplot(fig)


def afficher_interface():
    st.set_page_config(
        page_title="Audit Operator", layout="wide"
    )


    st.sidebar.markdown("""
    <div style="padding: 8px 0 18px 0;">
        <div style="font-size: 11px; letter-spacing: 3px; color: #72c9e4;">
            Adam Khazani Ahsaini
        </div>
        <div style="font-size: 21px; font-weight: 600; color: #e7eaed;">
            AUDIT OPERATOR
        </div>
        <div style="font-size: 10px; letter-spacing: 1px; color: #89959d;">
            ÉVALUATION DE MODÈLES PRÉDICTIFS
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    menu = st.sidebar.radio(
        "WORKSPACE",
        ["Audit Comparatif", "Audit Unitaire", "Config"],
        label_visibility="visible"
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("ACTIVE MARKET")
    st.sidebar.markdown(f"**{CONFIG['ticker']}**")

    date_fin = CONFIG["end_date"] or datetime.date.today()
    st.sidebar.caption(
        f"{CONFIG['start_date']} — {date_fin}"
    )



    st.markdown("""
    <style>
    /* =========================================
       AUDIT OPERATOR — INSTITUTIONAL UI
       ========================================= */

    :root {
        --background: #111519;
        --sidebar: #0d1216;
        --surface: #181f25;
        --surface-hover: #202a31;
        --border: #2d3942;
        --text: #e7edf2;
        --muted: #91a0ac;
        --accent: #70c9c0;
        --accent-hover: #8cddd3;
        --danger: #ef777e;
        --success: #75c9a5;
    }

    /* Fond général */
    .stApp {
        background: var(--background);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(17, 21, 25, 0.95);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    /* Typographie */
    html, body, [class*="css"] {
        font-family: Inter, -apple-system, BlinkMacSystemFont,
                     "Segoe UI", sans-serif;
    }

    h1, h2, h3, h4 {
        color: var(--text) !important;
        font-weight: 600 !important;
        letter-spacing: -0.035em;
    }

    h1 {
        font-size: 1.9rem !important;
        line-height: 1.3 !important;
    }

    h2 {
        font-size: 1.4rem !important;
        margin-top: 1.8rem !important;
    }

    h3 {
        font-size: 1.08rem !important;
    }

    p, label, li {
        color: var(--text);
    }

    [data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
        font-size: 0.82rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--sidebar);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        line-height: 1.55;
    }

    [data-testid="stSidebar"] hr {
        margin: 1.4rem 0;
        border-color: var(--border) !important;
    }

    /* Navigation radio */
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.35rem;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 0.45rem 0.6rem;
        border-radius: 6px;
        transition: background 0.15s ease;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: var(--surface-hover);
    }

    /* Cartes et conteneurs */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface);
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        transition: border-color 0.18s ease;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #43545e !important;
    }

    /* Métriques */
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.1rem;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-size: 0.8rem !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--accent) !important;
        font-size: 1.65rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.78rem !important;
    }

    /* Champs de saisie */
    .stTextInput input,
    .stNumberInput input,
    .stDateInput input,
    .stTextArea textarea {
        background: #12181d !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 7px !important;
    }

    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stDateInput input:focus,
    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }

    /* Boutons */
    .stButton button {
        border-radius: 7px;
        border: 1px solid var(--border);
        background: var(--surface);
        color: var(--text);
        font-weight: 500;
        transition: all 0.18s ease;
    }

    .stButton button:hover {
        border-color: var(--accent);
        color: var(--accent);
        background: var(--surface-hover);
    }

    .stButton button[kind="primary"] {
        background: var(--accent);
        color: #101819;
        border-color: var(--accent);
        font-weight: 650;
    }

    .stButton button[kind="primary"]:hover {
        background: var(--accent-hover);
        border-color: var(--accent-hover);
        color: #101819;
    }

    /* Cases à cocher */
    [data-testid="stCheckbox"] label {
        font-size: 0.84rem;
    }

    /* Import de fichiers */
    [data-testid="stFileUploader"] section {
        background: #131a1f;
        border: 1px dashed #40505a;
        border-radius: 8px;
        padding: 0.6rem;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: var(--accent);
    }

    [data-testid="stFileUploader"] button {
        border-radius: 6px;
    }

    /* Tableaux */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 9px;
        overflow: hidden;
    }

    /* Messages */
    [data-testid="stAlert"] {
        border-radius: 8px;
    }

    /* Barres de progression */
    [data-testid="stProgressBar"] > div {
        background: var(--surface-hover);
        border-radius: 99px;
    }

    [data-testid="stProgressBar"] > div > div {
        background: var(--accent);
        border-radius: 99px;
    }

    /* Séparateurs */
    hr {
        border-color: var(--border) !important;
    }

    /* Mobile */
    @media (max-width: 768px) {
        .block-container {
            padding: 1.2rem 1rem 2rem;
        }

        h1 {
            font-size: 1.55rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)


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