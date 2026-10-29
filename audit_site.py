import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from main import CONFIG, run_backtest, stock_getter
from streamlit.web import cli as stcli

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

    with st.expander("Ajouter un nouveau modèle Custom", expanded=False):
        custom_name = st.text_input(
            "Nom du modèle Custom :",
            placeholder="ex: RiskMetrics_2026, Proprietary_GARCH...",
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            file_var = st.file_uploader(
                "Série VaR", type=["csv", "xlsx", "parquet"], key="add_var"
            )
        with c2:
            file_es = st.file_uploader(
                "Série ES", type=["csv", "xlsx", "parquet"], key="add_es"
            )
        with c3:
            file_vol = st.file_uploader(
                "Série Volatilité",
                type=["csv", "xlsx", "parquet"],
                key="add_vol",
            )

        if st.button("Enregistrer ce modèle Custom"):
            clean_name = custom_name.strip()
            if not clean_name:
                st.error("Veuillez saisir un nom pour le modèle Custom.")
            elif clean_name in st.session_state["custom_models"]:
                st.error("Un modèle porte déjà ce nom.")
            elif not (file_var and file_es and file_vol):
                st.error("Veuillez importer les 3 fichiers requis.")
            else:
                st.session_state["custom_models"][clean_name] = {
                    "var": parse_uploaded_file(file_var),
                    "es": parse_uploaded_file(file_es),
                    "vol": parse_uploaded_file(file_vol),
                }
                st.success(f"Modèle Custom '{clean_name}' ajouté.")
                st.rerun()

    if st.session_state["custom_models"]:
        st.markdown("##### Modèles Custom en mémoire")
        cols_custom_select = {}
        for c_name in list(st.session_state["custom_models"].keys()):
            col_chk, col_del = st.columns([5, 1])
            with col_chk:
                is_selected = st.checkbox(
                    f"Inclure **{c_name}**", value=True, key=f"chk_{c_name}"
                )
                cols_custom_select[c_name] = is_selected
            with col_del:
                if st.button("Supprimer", key=f"del_{c_name}"):
                    del st.session_state["custom_models"][c_name]
                    st.rerun()

        return cols_custom_select
    return {}


def page_audit_comparatif():
    st.title("Audit Comparatif du Risque")

    st.subheader("1. Modèles Standards")
    col_chk1, col_chk2, col_chk3 = st.columns(3)
    with col_chk1:
        c_hist = st.checkbox(STANDARD_MODELS["historical"], value=True)
    with col_chk2:
        c_gt = st.checkbox(STANDARD_MODELS["garch_t"], value=True)
    with col_chk3:
        c_gjr = st.checkbox(STANDARD_MODELS["gjr_skewt"], value=True)

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
            df_display = df_summary.astype(str)
            st.dataframe(df_display, width="stretch")

            afficher_synthese_comparative(df_summary)


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
        df_display = df_summary.astype(str)
        st.dataframe(df_display, width="stretch")

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

    st.sidebar.title("Audit Operator")
    menu = st.sidebar.radio(
        "Navigation :", ["Audit Comparatif", "Audit Unitaire"]
    )

    if menu == "Audit Comparatif":
        page_audit_comparatif()
    elif menu == "Audit Unitaire":
        page_audit_unitaire()


if __name__ == "__main__":
    if not st.runtime.exists():
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
    else:
        afficher_interface()