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

# ---------------------------------------------------------------------------
# Internationalisation (FR / EN / ES / ZH / DE)
# ---------------------------------------------------------------------------

STANDARD_MODEL_CODES = ("historical", "garch_t", "gjr_skewt")

LANG_OPTIONS = {
    "fr": "Français",
    "en": "English",
    "es": "Español",
    "zh": "中文",
    "de": "Deutsch",
}

TR = {
    "fr": {
        # Models
        "model.historical": "Historique (HS)",
        "model.historical.desc": "Quantiles empiriques",
        "model.garch_t": "GARCH-t",
        "model.garch_t.desc": "Volatilité conditionnelle · Student-t",
        "model.gjr_skewt": "GJR-GARCH (Skew-t)",
        "model.gjr_skewt.desc": "Volatilité asymétrique · Skew-t",
        # Metrics
        "metric.basel_zone_actu": "Zone de Bâle (actuelle)",
        "metric.exceptions": "Exceptions VaR",
        "metric.exc_rate": "Taux d'exceptions (%)",
        "metric.obs_tested": "Observations testées",
        "metric.kupiec_pval": "Kupiec POF · p-value",
        "metric.christof_cc_pval": "Christoffersen CC · p-value",
        "metric.dq_pval": "Dynamic Quantile · p-value",
        "metric.christof_ind_pval": "Christoffersen IND · p-value",
        "metric.es_mf_pval": "ES McNeil-Frey · p-value",
        "metric.acerbi_z1_stat": "Acerbi-Szekely Z1 · statistique",
        "metric.acerbi_z1_pval": "Acerbi-Szekely Z1 · p-value",
        "metric.tail_resid_mean": "Moyenne des résidus de queue",
        "metric.conv_warnings": "Alertes de convergence",
        "metric.exc_2008_crisis": "Exceptions · crise 2008",
        # Status
        "status.pos": "Valide",
        "status.neg": "Erreur",
        "status.warn": "Attention",
        "status.info": "Info",
        # Charts
        "chart.returns": "Rendements",
        "chart.var": "VaR {conf}",
        "chart.es": "ES (FRTB)",
        "chart.exceptions": "Exceptions",
        "chart.var_model": "VaR {conf} · {name}",
        "chart.unavailable_series": "Séries VaR / rendements indisponibles pour ce modèle.",
        # Headers
        "header.validation.kicker": "Validation de modèles",
        "header.comparative.title": "Audit comparatif",
        "header.comparative.subtitle": "Backtest simultané de plusieurs modèles de VaR / ES sur un même instrument.",
        "header.unitaire.title": "Audit unitaire",
        "header.unitaire.subtitle": "Backtest détaillé d'un modèle : exceptions, tests statistiques, zone de Bâle.",
        "header.config.kicker": "Paramétrage",
        "header.config.title": "Configuration",
        "header.config.subtitle": "Instrument, période d'analyse et paramètres de backtest.",
        # Sections
        "sec.reference_models": "Modèles de référence",
        "sec.reference_models.desc": "Méthodologies à comparer",
        "sec.custom_models": "Modèles custom",
        "sec.custom_models.desc": "Séries de risque externes",
        "sec.execution": "Exécution",
        "sec.results": "Résultats",
        "sec.synthesis": "Synthèse",
        "sec.synthesis.desc": "Meilleur modèle par critère d'audit",
        "sec.compared_var": "VaR comparées",
        "sec.compared_var.desc": "Seuils VaR {conf} superposés aux rendements",
        "sec.metrics_breakdown": "Décomposition des métriques",
        "sec.metrics_breakdown.desc": "Toutes les statistiques du backtest",
        "sec.model_selection": "Sélection du modèle",
        "sec.analysis_period": "Période d'analyse",
        "sec.instrument": "Instrument",
        "sec.backtest_params": "Paramètres de backtest",
        "sec.backtest_params.desc": "Lecture seule · définis dans main.CONFIG",
        # UI
        "ui.include_in_audit": "Inclure dans l'audit",
        "ui.run_comparative": "Lancer l'audit comparatif",
        "ui.run_unit": "Lancer l'audit unitaire",
        "ui.select_model": "Sélectionnez le modèle :",
        "ui.metric": "Métrique",
        "ui.code": "Code",
        "ui.parameter": "Paramètre",
        "ui.value": "Valeur",
        "ui.best_model": "Meilleur modèle",
        "ui.criterion": "Critère",
        "ui.test": "Test",
        "ui.indicator": "Indicateur",
        "ui.import_model": "Importer un modèle externe",
        "ui.model_name": "Nom du modèle",
        "ui.model_name.ph": "ex. RiskMetrics_2026",
        "ui.var_series": "A · Série VaR",
        "ui.es_series": "B · Série Expected Shortfall",
        "ui.vol_series": "C · Série de volatilité",
        "ui.upload_caption": "Formats acceptés : CSV, XLSX, Parquet · index = dates, première colonne = valeurs.",
        "ui.save_model": "Enregistrer le modèle",
        "ui.saved_models.select": "Modèles enregistrés · sélection pour l'audit",
        "ui.saved_models": "Modèles enregistrés",
        "ui.external_model_desc": "Modèle externe · VaR / ES / Vol",
        "ui.remove": "Retirer",
        "ui.start_date": "Date de début (AAAA-MM-JJ)",
        "ui.end_date": "Date de fin (AAAA-MM-JJ, ou None)",
        "ui.ticker": "Ticker (Yahoo Finance)",
        "ui.colors_caption": "Vert : H₀ non rejetée (p > 5 %) · Rouge : H₀ rejetée (p ≤ 5 %).",
        "ui.valid_period": "Période valide.",
        # Ribbon
        "ribbon.instrument": "Instrument",
        "ribbon.period": "Période",
        "ribbon.window": "Fenêtre",
        "ribbon.refit": "Refit",
        "ribbon.alpha_var": "α VaR",
        "ribbon.alpha_es": "α ES",
        "ribbon.days": "j",
        # KPI
        "kpi.basel": "Zone de Bâle",
        "kpi.basel.sub": "Feu réglementaire actuel",
        "kpi.var_exc": "Exceptions VaR",
        "kpi.exc_rate": "Taux d'exceptions",
        "kpi.kupiec": "Kupiec POF · p-value",
        "kpi.h0_not_rejected": "H₀ non rejetée",
        "kpi.h0_rejected": "H₀ rejetée",
        "kpi.target": "cible ≈ {n}",
        "kpi.varcaption": "Backtest VaR {conf} & Expected Shortfall",
        # Synthesis
        "synth.var_accuracy": "Précision de la VaR",
        "synth.exc_indep": "Indépendance des exceptions",
        "synth.dynamic_quality": "Qualité dynamique",
        "synth.exc_calibration": "Calibration du taux d'exceptions",
        "synth.deviation_target": "Écart à la cible",
        "synth.obs_target": "{n} obs. · cible ≈ {t}",
        "synth.unavailable": "Synthèse automatique indisponible pour cette sélection.",
        # Sidebar
        "sb.tagline": "Validation des modèles<br>de VaR et d'Expected Shortfall",
        "sb.navigation": "Navigation",
        "sb.menu.comparative": "Audit Comparatif",
        "sb.menu.unitaire": "Audit Unitaire",
        "sb.menu.config": "Config",
        "sb.dark": "Thème sombre",
        "sb.language": "Langue",
        "sb.active_market": "Marché actif",
        "sb.footer": ("Données : Yahoo Finance<br>"
                      "VaR : Kupiec · Christoffersen · DQ<br>"
                      "ES : McNeil-Frey · Acerbi-Szekely<br>"
                      "Cadre : Bâle / FRTB"),
        # Zones
        "zone.VERTE": "VERTE",
        "zone.JAUNE": "JAUNE",
        "zone.ROUGE": "ROUGE",
        # Errors / progress
        "err.select_one_model": "Veuillez sélectionner au moins un modèle à auditer.",
        "err.ticker_required": "Veuillez saisir un ticker.",
        "err.ticker_not_found": "Ticker introuvable : « {symbol} ». Le ticker précédent est conservé.",
        "ok.ticker_valid": "Ticker valide : {symbol}",
        "err.date_format": "Format invalide, utilisez AAAA-MM-JJ. Les dates précédentes sont conservées.",
        "err.date_order": "La date de fin doit être postérieure à la date de début. Les dates précédentes sont conservées.",
        "err.period_too_short": "Période trop courte : ~{n} jours de bourse pour une fenêtre de {w}. Les dates précédentes sont conservées.",
        "err.download_failed": "Téléchargement des données impossible : {e}",
        "err.period_too_short_obs": "Période trop courte : {n} observations pour une fenêtre de {w} (minimum {m}).",
        "progress.computing": "Calcul du modèle « {name} » en cours",
        "progress.done": "Modèle « {name} » terminé",
        # Upload
        "upload.empty": "{kind} : fichier vide.",
        "upload.bad_format": "{kind} : format non reconnu (une seule colonne attendue).",
        "upload.dup_dates": "{kind} : dates dupliquées dans l'index.",
        "upload.all_nan": "{kind} : série entièrement NaN.",
        "upload.non_finite": "{kind} : contient des valeurs non finies.",
        # Custom
        "custom.name_required": "Veuillez saisir un nom de modèle.",
        "custom.name_exists": "Un modèle portant ce nom existe déjà.",
        "custom.all_required": "Veuillez importer les trois séries requises.",
        "custom.saved": "Modèle « {name} » enregistré.",
        "custom.var_kind": "VaR",
        "custom.es_kind": "ES",
        "custom.vol_kind": "Volatilité",
        # Config params
        "cfg.window_size": "Fenêtre d'estimation (jours)",
        "cfg.refit_freq": "Fréquence de réestimation (jours)",
        "cfg.alpha_var": "Niveau α VaR",
        "cfg.alpha_es": "Niveau α ES",
        "cfg.lags_dq": "Retards du test DQ",
    },
    "en": {
        # Models
        "model.historical": "Historical (HS)",
        "model.historical.desc": "Empirical quantiles",
        "model.garch_t": "GARCH-t",
        "model.garch_t.desc": "Conditional volatility · Student-t",
        "model.gjr_skewt": "GJR-GARCH (Skew-t)",
        "model.gjr_skewt.desc": "Asymmetric volatility · Skew-t",
        # Metrics
        "metric.basel_zone_actu": "Basel zone (current)",
        "metric.exceptions": "VaR exceptions",
        "metric.exc_rate": "Exception rate (%)",
        "metric.obs_tested": "Observations tested",
        "metric.kupiec_pval": "Kupiec POF · p-value",
        "metric.christof_cc_pval": "Christoffersen CC · p-value",
        "metric.dq_pval": "Dynamic Quantile · p-value",
        "metric.christof_ind_pval": "Christoffersen IND · p-value",
        "metric.es_mf_pval": "ES McNeil-Frey · p-value",
        "metric.acerbi_z1_stat": "Acerbi-Szekely Z1 · statistic",
        "metric.acerbi_z1_pval": "Acerbi-Szekely Z1 · p-value",
        "metric.tail_resid_mean": "Mean tail residuals",
        "metric.conv_warnings": "Convergence warnings",
        "metric.exc_2008_crisis": "Exceptions · 2008 crisis",
        # Status
        "status.pos": "Valid",
        "status.neg": "Error",
        "status.warn": "Warning",
        "status.info": "Info",
        # Charts
        "chart.returns": "Returns",
        "chart.var": "VaR {conf}",
        "chart.es": "ES (FRTB)",
        "chart.exceptions": "Exceptions",
        "chart.var_model": "VaR {conf} · {name}",
        "chart.unavailable_series": "VaR / returns series unavailable for this model.",
        # Headers
        "header.validation.kicker": "Model validation",
        "header.comparative.title": "Comparative Audit",
        "header.comparative.subtitle": "Simultaneous backtest of several VaR / ES models on the same instrument.",
        "header.unitaire.title": "Unit Audit",
        "header.unitaire.subtitle": "Detailed backtest of a model: exceptions, statistical tests, Basel zone.",
        "header.config.kicker": "Settings",
        "header.config.title": "Configuration",
        "header.config.subtitle": "Instrument, analysis period and backtest parameters.",
        # Sections
        "sec.reference_models": "Reference models",
        "sec.reference_models.desc": "Methodologies to compare",
        "sec.custom_models": "Custom models",
        "sec.custom_models.desc": "External risk series",
        "sec.execution": "Run",
        "sec.results": "Results",
        "sec.synthesis": "Summary",
        "sec.synthesis.desc": "Best model per audit criterion",
        "sec.compared_var": "Compared VaR",
        "sec.compared_var.desc": "VaR {conf} thresholds overlaid on returns",
        "sec.metrics_breakdown": "Metric breakdown",
        "sec.metrics_breakdown.desc": "All backtest statistics",
        "sec.model_selection": "Model selection",
        "sec.analysis_period": "Analysis period",
        "sec.instrument": "Instrument",
        "sec.backtest_params": "Backtest parameters",
        "sec.backtest_params.desc": "Read-only · defined in main.CONFIG",
        # UI
        "ui.include_in_audit": "Include in audit",
        "ui.run_comparative": "Run comparative audit",
        "ui.run_unit": "Run unit audit",
        "ui.select_model": "Select model:",
        "ui.metric": "Metric",
        "ui.code": "Code",
        "ui.parameter": "Parameter",
        "ui.value": "Value",
        "ui.best_model": "Best model",
        "ui.criterion": "Criterion",
        "ui.test": "Test",
        "ui.indicator": "Indicator",
        "ui.import_model": "Import external model",
        "ui.model_name": "Model name",
        "ui.model_name.ph": "e.g. RiskMetrics_2026",
        "ui.var_series": "A · VaR series",
        "ui.es_series": "B · Expected Shortfall series",
        "ui.vol_series": "C · Volatility series",
        "ui.upload_caption": "Accepted formats: CSV, XLSX, Parquet · index = dates, first column = values.",
        "ui.save_model": "Save model",
        "ui.saved_models.select": "Saved models · select for audit",
        "ui.saved_models": "Saved models",
        "ui.external_model_desc": "External model · VaR / ES / Vol",
        "ui.remove": "Remove",
        "ui.start_date": "Start date (YYYY-MM-DD)",
        "ui.end_date": "End date (YYYY-MM-DD, or None)",
        "ui.ticker": "Ticker (Yahoo Finance)",
        "ui.colors_caption": "Green: H₀ not rejected (p > 5%) · Red: H₀ rejected (p ≤ 5%).",
        "ui.valid_period": "Valid period.",
        # Ribbon
        "ribbon.instrument": "Instrument",
        "ribbon.period": "Period",
        "ribbon.window": "Window",
        "ribbon.refit": "Refit",
        "ribbon.alpha_var": "α VaR",
        "ribbon.alpha_es": "α ES",
        "ribbon.days": "d",
        # KPI
        "kpi.basel": "Basel zone",
        "kpi.basel.sub": "Current regulatory traffic light",
        "kpi.var_exc": "VaR exceptions",
        "kpi.exc_rate": "Exception rate",
        "kpi.kupiec": "Kupiec POF · p-value",
        "kpi.h0_not_rejected": "H₀ not rejected",
        "kpi.h0_rejected": "H₀ rejected",
        "kpi.target": "target ≈ {n}",
        "kpi.varcaption": "Backtest VaR {conf} & Expected Shortfall",
        # Synthesis
        "synth.var_accuracy": "VaR accuracy",
        "synth.exc_indep": "Exception independence",
        "synth.dynamic_quality": "Dynamic quality",
        "synth.exc_calibration": "Exception rate calibration",
        "synth.deviation_target": "Deviation from target",
        "synth.obs_target": "{n} obs. · target ≈ {t}",
        "synth.unavailable": "Automatic summary unavailable for this selection.",
        # Sidebar
        "sb.tagline": "Model validation<br>of VaR and Expected Shortfall",
        "sb.navigation": "Navigation",
        "sb.menu.comparative": "Comparative Audit",
        "sb.menu.unitaire": "Unit Audit",
        "sb.menu.config": "Config",
        "sb.dark": "Dark theme",
        "sb.language": "Language",
        "sb.active_market": "Active market",
        "sb.footer": ("Data: Yahoo Finance<br>"
                      "VaR: Kupiec · Christoffersen · DQ<br>"
                      "ES: McNeil-Frey · Acerbi-Szekely<br>"
                      "Framework: Basel / FRTB"),
        # Zones
        "zone.VERTE": "GREEN",
        "zone.JAUNE": "YELLOW",
        "zone.ROUGE": "RED",
        # Errors / progress
        "err.select_one_model": "Please select at least one model to audit.",
        "err.ticker_required": "Please enter a ticker.",
        "err.ticker_not_found": "Ticker not found: \"{symbol}\". Previous ticker kept.",
        "ok.ticker_valid": "Valid ticker: {symbol}",
        "err.date_format": "Invalid format, use YYYY-MM-DD. Previous dates kept.",
        "err.date_order": "End date must be after start date. Previous dates kept.",
        "err.period_too_short": "Period too short: ~{n} trading days for a {w}-day window. Previous dates kept.",
        "err.download_failed": "Data download failed: {e}",
        "err.period_too_short_obs": "Period too short: {n} observations for a {w}-day window (minimum {m}).",
        "progress.computing": "Computing model \"{name}\"",
        "progress.done": "Model \"{name}\" done",
        # Upload
        "upload.empty": "{kind}: empty file.",
        "upload.bad_format": "{kind}: unrecognized format (single column expected).",
        "upload.dup_dates": "{kind}: duplicate dates in index.",
        "upload.all_nan": "{kind}: series is entirely NaN.",
        "upload.non_finite": "{kind}: contains non-finite values.",
        # Custom
        "custom.name_required": "Please enter a model name.",
        "custom.name_exists": "A model with this name already exists.",
        "custom.all_required": "Please upload all three required series.",
        "custom.saved": "Model \"{name}\" saved.",
        "custom.var_kind": "VaR",
        "custom.es_kind": "ES",
        "custom.vol_kind": "Volatility",
        # Config params
        "cfg.window_size": "Estimation window (days)",
        "cfg.refit_freq": "Refit frequency (days)",
        "cfg.alpha_var": "α VaR level",
        "cfg.alpha_es": "α ES level",
        "cfg.lags_dq": "DQ test lags",
    },
    "es": {
        # Models
        "model.historical": "Histórico (HS)",
        "model.historical.desc": "Cuantiles empíricos",
        "model.garch_t": "GARCH-t",
        "model.garch_t.desc": "Volatilidad condicional · Student-t",
        "model.gjr_skewt": "GJR-GARCH (Skew-t)",
        "model.gjr_skewt.desc": "Volatilidad asimétrica · Skew-t",
        # Metrics
        "metric.basel_zone_actu": "Zona de Basilea (actual)",
        "metric.exceptions": "Excepciones VaR",
        "metric.exc_rate": "Tasa de excepciones (%)",
        "metric.obs_tested": "Observaciones probadas",
        "metric.kupiec_pval": "Kupiec POF · valor-p",
        "metric.christof_cc_pval": "Christoffersen CC · valor-p",
        "metric.dq_pval": "Cuantil Dinámico · valor-p",
        "metric.christof_ind_pval": "Christoffersen IND · valor-p",
        "metric.es_mf_pval": "ES McNeil-Frey · valor-p",
        "metric.acerbi_z1_stat": "Acerbi-Szekely Z1 · estadístico",
        "metric.acerbi_z1_pval": "Acerbi-Szekely Z1 · valor-p",
        "metric.tail_resid_mean": "Media de residuos de cola",
        "metric.conv_warnings": "Advertencias de convergencia",
        "metric.exc_2008_crisis": "Excepciones · crisis 2008",
        # Status
        "status.pos": "Válido",
        "status.neg": "Error",
        "status.warn": "Atención",
        "status.info": "Info",
        # Charts
        "chart.returns": "Rendimientos",
        "chart.var": "VaR {conf}",
        "chart.es": "ES (FRTB)",
        "chart.exceptions": "Excepciones",
        "chart.var_model": "VaR {conf} · {name}",
        "chart.unavailable_series": "Series VaR / rendimientos no disponibles para este modelo.",
        # Headers
        "header.validation.kicker": "Validación de modelos",
        "header.comparative.title": "Auditoría comparativa",
        "header.comparative.subtitle": "Backtesting simultáneo de varios modelos de VaR / ES sobre el mismo instrumento.",
        "header.unitaire.title": "Auditoría unitaria",
        "header.unitaire.subtitle": "Backtesting detallado de un modelo: excepciones, pruebas estadísticas, zona de Basilea.",
        "header.config.kicker": "Configuración",
        "header.config.title": "Configuración",
        "header.config.subtitle": "Instrumento, período de análisis y parámetros de backtesting.",
        # Sections
        "sec.reference_models": "Modelos de referencia",
        "sec.reference_models.desc": "Metodologías a comparar",
        "sec.custom_models": "Modelos personalizados",
        "sec.custom_models.desc": "Series de riesgo externas",
        "sec.execution": "Ejecución",
        "sec.results": "Resultados",
        "sec.synthesis": "Síntesis",
        "sec.synthesis.desc": "Mejor modelo por criterio de auditoría",
        "sec.compared_var": "VaR comparadas",
        "sec.compared_var.desc": "Umbrales VaR {conf} superpuestos a los rendimientos",
        "sec.metrics_breakdown": "Desglose de métricas",
        "sec.metrics_breakdown.desc": "Todas las estadísticas del backtesting",
        "sec.model_selection": "Selección del modelo",
        "sec.analysis_period": "Período de análisis",
        "sec.instrument": "Instrumento",
        "sec.backtest_params": "Parámetros de backtesting",
        "sec.backtest_params.desc": "Solo lectura · definidos en main.CONFIG",
        # UI
        "ui.include_in_audit": "Incluir en la auditoría",
        "ui.run_comparative": "Ejecutar auditoría comparativa",
        "ui.run_unit": "Ejecutar auditoría unitaria",
        "ui.select_model": "Seleccione el modelo:",
        "ui.metric": "Métrica",
        "ui.code": "Código",
        "ui.parameter": "Parámetro",
        "ui.value": "Valor",
        "ui.best_model": "Mejor modelo",
        "ui.criterion": "Criterio",
        "ui.test": "Prueba",
        "ui.indicator": "Indicador",
        "ui.import_model": "Importar modelo externo",
        "ui.model_name": "Nombre del modelo",
        "ui.model_name.ph": "ej. RiskMetrics_2026",
        "ui.var_series": "A · Serie VaR",
        "ui.es_series": "B · Serie Expected Shortfall",
        "ui.vol_series": "C · Serie de volatilidad",
        "ui.upload_caption": "Formatos aceptados: CSV, XLSX, Parquet · índice = fechas, primera columna = valores.",
        "ui.save_model": "Guardar modelo",
        "ui.saved_models.select": "Modelos guardados · seleccionar para auditoría",
        "ui.saved_models": "Modelos guardados",
        "ui.external_model_desc": "Modelo externo · VaR / ES / Vol",
        "ui.remove": "Eliminar",
        "ui.start_date": "Fecha de inicio (AAAA-MM-DD)",
        "ui.end_date": "Fecha de fin (AAAA-MM-DD, o None)",
        "ui.ticker": "Ticker (Yahoo Finance)",
        "ui.colors_caption": "Verde: H₀ no rechazada (p > 5%) · Rojo: H₀ rechazada (p ≤ 5%).",
        "ui.valid_period": "Período válido.",
        # Ribbon
        "ribbon.instrument": "Instrumento",
        "ribbon.period": "Período",
        "ribbon.window": "Ventana",
        "ribbon.refit": "Refit",
        "ribbon.alpha_var": "α VaR",
        "ribbon.alpha_es": "α ES",
        "ribbon.days": "d",
        # KPI
        "kpi.basel": "Zona de Basilea",
        "kpi.basel.sub": "Semáforo regulatorio actual",
        "kpi.var_exc": "Excepciones VaR",
        "kpi.exc_rate": "Tasa de excepciones",
        "kpi.kupiec": "Kupiec POF · valor-p",
        "kpi.h0_not_rejected": "H₀ no rechazada",
        "kpi.h0_rejected": "H₀ rechazada",
        "kpi.target": "objetivo ≈ {n}",
        "kpi.varcaption": "Backtesting VaR {conf} & Expected Shortfall",
        # Synthesis
        "synth.var_accuracy": "Precisión de la VaR",
        "synth.exc_indep": "Independencia de las excepciones",
        "synth.dynamic_quality": "Calidad dinámica",
        "synth.exc_calibration": "Calibración de la tasa de excepciones",
        "synth.deviation_target": "Desviación del objetivo",
        "synth.obs_target": "{n} obs. · objetivo ≈ {t}",
        "synth.unavailable": "Síntesis automática no disponible para esta selección.",
        # Sidebar
        "sb.tagline": "Validación de modelos<br>de VaR y Expected Shortfall",
        "sb.navigation": "Navegación",
        "sb.menu.comparative": "Auditoría Comparativa",
        "sb.menu.unitaire": "Auditoría Unitaria",
        "sb.menu.config": "Config",
        "sb.dark": "Tema oscuro",
        "sb.language": "Idioma",
        "sb.active_market": "Mercado activo",
        "sb.footer": ("Datos: Yahoo Finance<br>"
                      "VaR: Kupiec · Christoffersen · DQ<br>"
                      "ES: McNeil-Frey · Acerbi-Szekely<br>"
                      "Marco: Basilea / FRTB"),
        # Zones
        "zone.VERTE": "VERDE",
        "zone.JAUNE": "AMARILLO",
        "zone.ROUGE": "ROJO",
        # Errors / progress
        "err.select_one_model": "Por favor, seleccione al menos un modelo para auditar.",
        "err.ticker_required": "Por favor, introduzca un ticker.",
        "err.ticker_not_found": "Ticker no encontrado: «{symbol}». Se conserva el ticker anterior.",
        "ok.ticker_valid": "Ticker válido: {symbol}",
        "err.date_format": "Formato inválido, utilice AAAA-MM-DD. Se conservan las fechas anteriores.",
        "err.date_order": "La fecha de fin debe ser posterior a la fecha de inicio. Se conservan las fechas anteriores.",
        "err.period_too_short": "Período demasiado corto: ~{n} días bursátiles para una ventana de {w}. Se conservan las fechas anteriores.",
        "err.download_failed": "Descarga de datos imposible: {e}",
        "err.period_too_short_obs": "Período demasiado corto: {n} observaciones para una ventana de {w} (mínimo {m}).",
        "progress.computing": "Calculando el modelo «{name}»",
        "progress.done": "Modelo «{name}» terminado",
        # Upload
        "upload.empty": "{kind}: archivo vacío.",
        "upload.bad_format": "{kind}: formato no reconocido (se espera una sola columna).",
        "upload.dup_dates": "{kind}: fechas duplicadas en el índice.",
        "upload.all_nan": "{kind}: serie completamente NaN.",
        "upload.non_finite": "{kind}: contiene valores no finitos.",
        # Custom
        "custom.name_required": "Por favor, introduzca un nombre de modelo.",
        "custom.name_exists": "Ya existe un modelo con este nombre.",
        "custom.all_required": "Por favor, importe las tres series requeridas.",
        "custom.saved": "Modelo «{name}» guardado.",
        "custom.var_kind": "VaR",
        "custom.es_kind": "ES",
        "custom.vol_kind": "Volatilidad",
        # Config params
        "cfg.window_size": "Ventana de estimación (días)",
        "cfg.refit_freq": "Frecuencia de reestimación (días)",
        "cfg.alpha_var": "Nivel α VaR",
        "cfg.alpha_es": "Nivel α ES",
        "cfg.lags_dq": "Rezagos de la prueba DQ",
    },
    "zh": {
        # Models
        "model.historical": "历史模拟法 (HS)",
        "model.historical.desc": "经验分位数",
        "model.garch_t": "GARCH-t",
        "model.garch_t.desc": "条件波动率 · Student-t",
        "model.gjr_skewt": "GJR-GARCH (Skew-t)",
        "model.gjr_skewt.desc": "非对称波动率 · Skew-t",
        # Metrics
        "metric.basel_zone_actu": "巴塞尔区域 (当前)",
        "metric.exceptions": "VaR 例外",
        "metric.exc_rate": "例外率 (%)",
        "metric.obs_tested": "测试观测数",
        "metric.kupiec_pval": "Kupiec POF · p 值",
        "metric.christof_cc_pval": "Christoffersen CC · p 值",
        "metric.dq_pval": "动态分位数 · p 值",
        "metric.christof_ind_pval": "Christoffersen IND · p 值",
        "metric.es_mf_pval": "ES McNeil-Frey · p 值",
        "metric.acerbi_z1_stat": "Acerbi-Szekely Z1 · 统计量",
        "metric.acerbi_z1_pval": "Acerbi-Szekely Z1 · p 值",
        "metric.tail_resid_mean": "尾部残差均值",
        "metric.conv_warnings": "收敛警告",
        "metric.exc_2008_crisis": "例外 · 2008 年危机",
        # Status
        "status.pos": "有效",
        "status.neg": "错误",
        "status.warn": "警告",
        "status.info": "信息",
        # Charts
        "chart.returns": "收益率",
        "chart.var": "VaR {conf}",
        "chart.es": "ES (FRTB)",
        "chart.exceptions": "例外",
        "chart.var_model": "VaR {conf} · {name}",
        "chart.unavailable_series": "此模型的 VaR / 收益率序列不可用。",
        # Headers
        "header.validation.kicker": "模型验证",
        "header.comparative.title": "比较审计",
        "header.comparative.subtitle": "在同一工具上同时回测多个 VaR / ES 模型。",
        "header.unitaire.title": "单元审计",
        "header.unitaire.subtitle": "单一模型的详细回测：例外、统计检验、巴塞尔区域。",
        "header.config.kicker": "设置",
        "header.config.title": "配置",
        "header.config.subtitle": "工具、分析期间和回测参数。",
        # Sections
        "sec.reference_models": "参考模型",
        "sec.reference_models.desc": "待比较的方法",
        "sec.custom_models": "自定义模型",
        "sec.custom_models.desc": "外部风险序列",
        "sec.execution": "运行",
        "sec.results": "结果",
        "sec.synthesis": "摘要",
        "sec.synthesis.desc": "按审计标准的最佳模型",
        "sec.compared_var": "VaR 比较",
        "sec.compared_var.desc": "VaR {conf} 阈值与收益率叠加",
        "sec.metrics_breakdown": "指标分解",
        "sec.metrics_breakdown.desc": "所有回测统计量",
        "sec.model_selection": "模型选择",
        "sec.analysis_period": "分析期间",
        "sec.instrument": "工具",
        "sec.backtest_params": "回测参数",
        "sec.backtest_params.desc": "只读 · 定义于 main.CONFIG",
        # UI
        "ui.include_in_audit": "纳入审计",
        "ui.run_comparative": "运行比较审计",
        "ui.run_unit": "运行单元审计",
        "ui.select_model": "选择模型：",
        "ui.metric": "指标",
        "ui.code": "代码",
        "ui.parameter": "参数",
        "ui.value": "值",
        "ui.best_model": "最佳模型",
        "ui.criterion": "标准",
        "ui.test": "检验",
        "ui.indicator": "指标",
        "ui.import_model": "导入外部模型",
        "ui.model_name": "模型名称",
        "ui.model_name.ph": "例如 RiskMetrics_2026",
        "ui.var_series": "A · VaR 序列",
        "ui.es_series": "B · Expected Shortfall 序列",
        "ui.vol_series": "C · 波动率序列",
        "ui.upload_caption": "接受的格式：CSV、XLSX、Parquet · 索引 = 日期，第一列 = 值。",
        "ui.save_model": "保存模型",
        "ui.saved_models.select": "已保存的模型 · 选择以进行审计",
        "ui.saved_models": "已保存的模型",
        "ui.external_model_desc": "外部模型 · VaR / ES / Vol",
        "ui.remove": "移除",
        "ui.start_date": "开始日期 (YYYY-MM-DD)",
        "ui.end_date": "结束日期 (YYYY-MM-DD，或 None)",
        "ui.ticker": "股票代码 (Yahoo Finance)",
        "ui.colors_caption": "绿色：H₀ 未被拒绝 (p > 5%) · 红色：H₀ 被拒绝 (p ≤ 5%)。",
        "ui.valid_period": "有效期间。",
        # Ribbon
        "ribbon.instrument": "工具",
        "ribbon.period": "期间",
        "ribbon.window": "窗口",
        "ribbon.refit": "重拟合",
        "ribbon.alpha_var": "α VaR",
        "ribbon.alpha_es": "α ES",
        "ribbon.days": "天",
        # KPI
        "kpi.basel": "巴塞尔区域",
        "kpi.basel.sub": "当前监管交通灯",
        "kpi.var_exc": "VaR 例外",
        "kpi.exc_rate": "例外率",
        "kpi.kupiec": "Kupiec POF · p 值",
        "kpi.h0_not_rejected": "H₀ 未被拒绝",
        "kpi.h0_rejected": "H₀ 被拒绝",
        "kpi.target": "目标 ≈ {n}",
        "kpi.varcaption": "回测 VaR {conf} & Expected Shortfall",
        # Synthesis
        "synth.var_accuracy": "VaR 准确性",
        "synth.exc_indep": "例外独立性",
        "synth.dynamic_quality": "动态质量",
        "synth.exc_calibration": "例外率校准",
        "synth.deviation_target": "与目标的偏差",
        "synth.obs_target": "{n} 个观测 · 目标 ≈ {t}",
        "synth.unavailable": "此选择无法自动生成摘要。",
        # Sidebar
        "sb.tagline": "VaR 与 Expected Shortfall<br>模型验证",
        "sb.navigation": "导航",
        "sb.menu.comparative": "比较审计",
        "sb.menu.unitaire": "单元审计",
        "sb.menu.config": "配置",
        "sb.dark": "深色主题",
        "sb.language": "语言",
        "sb.active_market": "活跃市场",
        "sb.footer": ("数据：Yahoo Finance<br>"
                      "VaR：Kupiec · Christoffersen · DQ<br>"
                      "ES：McNeil-Frey · Acerbi-Szekely<br>"
                      "框架：巴塞尔 / FRTB"),
        # Zones
        "zone.VERTE": "绿色",
        "zone.JAUNE": "黄色",
        "zone.ROUGE": "红色",
        # Errors / progress
        "err.select_one_model": "请选择至少一个要审计的模型。",
        "err.ticker_required": "请输入股票代码。",
        "err.ticker_not_found": "未找到股票代码：\"{symbol}\"。保留之前的代码。",
        "ok.ticker_valid": "有效代码：{symbol}",
        "err.date_format": "格式无效，请使用 YYYY-MM-DD。保留之前的日期。",
        "err.date_order": "结束日期必须晚于开始日期。保留之前的日期。",
        "err.period_too_short": "期间太短：约 {n} 个交易日，窗口为 {w} 天。保留之前的日期。",
        "err.download_failed": "数据下载失败：{e}",
        "err.period_too_short_obs": "期间太短：{n} 个观测，窗口为 {w} 天（最小 {m}）。",
        "progress.computing": "正在计算模型「{name}」",
        "progress.done": "模型「{name}」完成",
        # Upload
        "upload.empty": "{kind}：文件为空。",
        "upload.bad_format": "{kind}：格式无法识别（应为一列）。",
        "upload.dup_dates": "{kind}：索引中存在重复日期。",
        "upload.all_nan": "{kind}：序列全为 NaN。",
        "upload.non_finite": "{kind}：包含非有限值。",
        # Custom
        "custom.name_required": "请输入模型名称。",
        "custom.name_exists": "已存在同名模型。",
        "custom.all_required": "请导入所有三个必需的序列。",
        "custom.saved": "模型「{name}」已保存。",
        "custom.var_kind": "VaR",
        "custom.es_kind": "ES",
        "custom.vol_kind": "波动率",
        # Config params
        "cfg.window_size": "估计窗口（天）",
        "cfg.refit_freq": "重拟合频率（天）",
        "cfg.alpha_var": "α VaR 水平",
        "cfg.alpha_es": "α ES 水平",
        "cfg.lags_dq": "DQ 检验滞后",
    },
    "de": {
        # Models
        "model.historical": "Historisch (HS)",
        "model.historical.desc": "Empirische Quantile",
        "model.garch_t": "GARCH-t",
        "model.garch_t.desc": "Bedingte Volatilität · Student-t",
        "model.gjr_skewt": "GJR-GARCH (Skew-t)",
        "model.gjr_skewt.desc": "Asymmetrische Volatilität · Skew-t",
        # Metrics
        "metric.basel_zone_actu": "Basel-Zone (aktuell)",
        "metric.exceptions": "VaR-Ausnahmen",
        "metric.exc_rate": "Ausnahmenrate (%)",
        "metric.obs_tested": "Getestete Beobachtungen",
        "metric.kupiec_pval": "Kupiec POF · p-Wert",
        "metric.christof_cc_pval": "Christoffersen CC · p-Wert",
        "metric.dq_pval": "Dynamisches Quantil · p-Wert",
        "metric.christof_ind_pval": "Christoffersen IND · p-Wert",
        "metric.es_mf_pval": "ES McNeil-Frey · p-Wert",
        "metric.acerbi_z1_stat": "Acerbi-Szekely Z1 · Statistik",
        "metric.acerbi_z1_pval": "Acerbi-Szekely Z1 · p-Wert",
        "metric.tail_resid_mean": "Mittlere Tail-Residuen",
        "metric.conv_warnings": "Konvergenzwarnungen",
        "metric.exc_2008_crisis": "Ausnahmen · Krise 2008",
        # Status
        "status.pos": "Gültig",
        "status.neg": "Fehler",
        "status.warn": "Achtung",
        "status.info": "Info",
        # Charts
        "chart.returns": "Renditen",
        "chart.var": "VaR {conf}",
        "chart.es": "ES (FRTB)",
        "chart.exceptions": "Ausnahmen",
        "chart.var_model": "VaR {conf} · {name}",
        "chart.unavailable_series": "VaR-/Renditereihen für dieses Modell nicht verfügbar.",
        # Headers
        "header.validation.kicker": "Modellvalidierung",
        "header.comparative.title": "Vergleichendes Audit",
        "header.comparative.subtitle": "Simultanes Backtesting mehrerer VaR-/ES-Modelle auf demselben Instrument.",
        "header.unitaire.title": "Einzel-Audit",
        "header.unitaire.subtitle": "Detailliertes Backtesting eines Modells: Ausnahmen, statistische Tests, Basel-Zone.",
        "header.config.kicker": "Einstellungen",
        "header.config.title": "Konfiguration",
        "header.config.subtitle": "Instrument, Analysezeitraum und Backtesting-Parameter.",
        # Sections
        "sec.reference_models": "Referenzmodelle",
        "sec.reference_models.desc": "Zu vergleichende Methoden",
        "sec.custom_models": "Benutzerdefinierte Modelle",
        "sec.custom_models.desc": "Externe Risikoreihen",
        "sec.execution": "Ausführung",
        "sec.results": "Ergebnisse",
        "sec.synthesis": "Zusammenfassung",
        "sec.synthesis.desc": "Bestes Modell pro Auditkriterium",
        "sec.compared_var": "Verglichene VaR",
        "sec.compared_var.desc": "VaR-{conf}-Schwellen überlagert auf Renditen",
        "sec.metrics_breakdown": "Metrik-Aufschlüsselung",
        "sec.metrics_breakdown.desc": "Alle Backtesting-Statistiken",
        "sec.model_selection": "Modellauswahl",
        "sec.analysis_period": "Analysezeitraum",
        "sec.instrument": "Instrument",
        "sec.backtest_params": "Backtesting-Parameter",
        "sec.backtest_params.desc": "Schreibgeschützt · definiert in main.CONFIG",
        # UI
        "ui.include_in_audit": "In Audit einbeziehen",
        "ui.run_comparative": "Vergleichendes Audit starten",
        "ui.run_unit": "Einzel-Audit starten",
        "ui.select_model": "Modell auswählen:",
        "ui.metric": "Metrik",
        "ui.code": "Code",
        "ui.parameter": "Parameter",
        "ui.value": "Wert",
        "ui.best_model": "Bestes Modell",
        "ui.criterion": "Kriterium",
        "ui.test": "Test",
        "ui.indicator": "Indikator",
        "ui.import_model": "Externes Modell importieren",
        "ui.model_name": "Modellname",
        "ui.model_name.ph": "z. B. RiskMetrics_2026",
        "ui.var_series": "A · VaR-Reihe",
        "ui.es_series": "B · Expected-Shortfall-Reihe",
        "ui.vol_series": "C · Volatilitätsreihe",
        "ui.upload_caption": "Akzeptierte Formate: CSV, XLSX, Parquet · Index = Datum, erste Spalte = Werte.",
        "ui.save_model": "Modell speichern",
        "ui.saved_models.select": "Gespeicherte Modelle · für Audit auswählen",
        "ui.saved_models": "Gespeicherte Modelle",
        "ui.external_model_desc": "Externes Modell · VaR / ES / Vol",
        "ui.remove": "Entfernen",
        "ui.start_date": "Startdatum (JJJJ-MM-TT)",
        "ui.end_date": "Enddatum (JJJJ-MM-TT oder None)",
        "ui.ticker": "Ticker (Yahoo Finance)",
        "ui.colors_caption": "Grün: H₀ nicht abgelehnt (p > 5 %) · Rot: H₀ abgelehnt (p ≤ 5 %).",
        "ui.valid_period": "Gültiger Zeitraum.",
        # Ribbon
        "ribbon.instrument": "Instrument",
        "ribbon.period": "Zeitraum",
        "ribbon.window": "Fenster",
        "ribbon.refit": "Refit",
        "ribbon.alpha_var": "α VaR",
        "ribbon.alpha_es": "α ES",
        "ribbon.days": "T",
        # KPI
        "kpi.basel": "Basel-Zone",
        "kpi.basel.sub": "Aktuelle regulatorische Ampel",
        "kpi.var_exc": "VaR-Ausnahmen",
        "kpi.exc_rate": "Ausnahmenrate",
        "kpi.kupiec": "Kupiec POF · p-Wert",
        "kpi.h0_not_rejected": "H₀ nicht abgelehnt",
        "kpi.h0_rejected": "H₀ abgelehnt",
        "kpi.target": "Ziel ≈ {n}",
        "kpi.varcaption": "Backtesting VaR {conf} & Expected Shortfall",
        # Synthesis
        "synth.var_accuracy": "VaR-Genauigkeit",
        "synth.exc_indep": "Unabhängigkeit der Ausnahmen",
        "synth.dynamic_quality": "Dynamische Qualität",
        "synth.exc_calibration": "Kalibrierung der Ausnahmenrate",
        "synth.deviation_target": "Abweichung vom Ziel",
        "synth.obs_target": "{n} Beob. · Ziel ≈ {t}",
        "synth.unavailable": "Automatische Zusammenfassung für diese Auswahl nicht verfügbar.",
        # Sidebar
        "sb.tagline": "Modellvalidierung<br>von VaR und Expected Shortfall",
        "sb.navigation": "Navigation",
        "sb.menu.comparative": "Vergleichendes Audit",
        "sb.menu.unitaire": "Einzel-Audit",
        "sb.menu.config": "Konfig",
        "sb.dark": "Dunkles Design",
        "sb.language": "Sprache",
        "sb.active_market": "Aktiver Markt",
        "sb.footer": ("Daten: Yahoo Finance<br>"
                      "VaR: Kupiec · Christoffersen · DQ<br>"
                      "ES: McNeil-Frey · Acerbi-Szekely<br>"
                      "Rahmen: Basel / FRTB"),
        # Zones
        "zone.VERTE": "GRÜN",
        "zone.JAUNE": "GELB",
        "zone.ROUGE": "ROT",
        # Errors / progress
        "err.select_one_model": "Bitte wählen Sie mindestens ein Modell zum Auditieren aus.",
        "err.ticker_required": "Bitte geben Sie einen Ticker ein.",
        "err.ticker_not_found": "Ticker nicht gefunden: „{symbol}“. Vorheriger Ticker beibehalten.",
        "ok.ticker_valid": "Gültiger Ticker: {symbol}",
        "err.date_format": "Ungültiges Format, verwenden Sie JJJJ-MM-TT. Vorherige Daten beibehalten.",
        "err.date_order": "Enddatum muss nach dem Startdatum liegen. Vorherige Daten beibehalten.",
        "err.period_too_short": "Zeitraum zu kurz: ~{n} Handelstage für ein {w}-Tage-Fenster. Vorherige Daten beibehalten.",
        "err.download_failed": "Daten-Download fehlgeschlagen: {e}",
        "err.period_too_short_obs": "Zeitraum zu kurz: {n} Beobachtungen für ein {w}-Tage-Fenster (Minimum {m}).",
        "progress.computing": "Berechne Modell „{name}“",
        "progress.done": "Modell „{name}“ fertig",
        # Upload
        "upload.empty": "{kind}: leere Datei.",
        "upload.bad_format": "{kind}: Format nicht erkannt (eine Spalte erwartet).",
        "upload.dup_dates": "{kind}: doppelte Daten im Index.",
        "upload.all_nan": "{kind}: Reihe vollständig NaN.",
        "upload.non_finite": "{kind}: enthält nicht-finite Werte.",
        # Custom
        "custom.name_required": "Bitte geben Sie einen Modellnamen ein.",
        "custom.name_exists": "Ein Modell mit diesem Namen existiert bereits.",
        "custom.all_required": "Bitte importieren Sie alle drei erforderlichen Reihen.",
        "custom.saved": "Modell „{name}“ gespeichert.",
        "custom.var_kind": "VaR",
        "custom.es_kind": "ES",
        "custom.vol_kind": "Volatilität",
        # Config params
        "cfg.window_size": "Schätzfenster (Tage)",
        "cfg.refit_freq": "Refit-Häufigkeit (Tage)",
        "cfg.alpha_var": "α-VaR-Niveau",
        "cfg.alpha_es": "α-ES-Niveau",
        "cfg.lags_dq": "DQ-Test-Lags",
    },
}


def t(key, **kw):
    """Translate a key using the current language. Falls back to FR, then to the key."""
    lang = st.session_state.get("lang", "fr")
    d = TR.get(lang) or TR["fr"]
    entry = d.get(key)
    if entry is None:
        entry = TR["fr"].get(key, key)
    if kw:
        try:
            return entry.format(**kw)
        except (KeyError, IndexError, ValueError):
            return entry
    return entry


def _metric_label(key):
    lang = st.session_state.get("lang", "fr")
    d = TR.get(lang) or TR["fr"]
    label = d.get(f"metric.{key}") or TR["fr"].get(f"metric.{key}")
    if label is None:
        return str(key).replace("_", " ").capitalize()
    return label


def _display_label(key):
    """Map an internal model identifier to its localized display label."""
    if key in STANDARD_MODEL_CODES:
        return t(f"model.{key}")
    return str(key)


MODEL_COLORS = []
C = {}


# ---------------------------------------------------------------------------
# Theme utilities (unchanged)
# ---------------------------------------------------------------------------

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
    mode = st.session_state.get("theme_mode", "light")
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


# ---------------------------------------------------------------------------
# Data / backtest helpers
# ---------------------------------------------------------------------------

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
        raise ValueError(t("upload.empty", kind=kind))
    if not isinstance(series, pd.Series):
        raise ValueError(t("upload.bad_format", kind=kind))
    if not series.index.is_monotonic_increasing:
        series = series.sort_index()
    if series.index.has_duplicates:
        raise ValueError(t("upload.dup_dates", kind=kind))
    if series.isna().all():
        raise ValueError(t("upload.all_nan", kind=kind))
    if not np.isfinite(series.dropna().values).all():
        raise ValueError(t("upload.non_finite", kind=kind))
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
        return False, t("err.download_failed", e=e)
    min_obs = cfg["window_size"] + CONFIG["lags_dq"] + 20
    if n <= min_obs:
        return False, t("err.period_too_short_obs",
                        n=n, w=cfg["window_size"], m=min_obs)
    return True, None


def backtester(selected_standards, selected_customs_dict, progress_callback=None):
    cfg = _cfg()
    tasks = []
    for code in STANDARD_MODEL_CODES:
        if code in selected_standards:
            tasks.append((code, code, None))
    for c_name, c_data in selected_customs_dict.items():
        tasks.append(("custom", c_name, c_data))

    total = len(tasks)
    results = []
    dict_outputs = {}

    for i, (m_type, key, c_data) in enumerate(tasks):
        pct_depart = int((i / total) * 100)
        if progress_callback:
            progress_callback(pct_depart,
                              t("progress.computing", name=_display_label(key)))

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

        dict_outputs[key] = res
        clean_res = {
            k: v for k, v in res.items()
            if k not in ["var_series", "es_frtb_series", "test_returns", "rolling_250",
                         "eta_hist", "lambda_hist", "refit_dates"]
        }
        clean_res["model"] = key
        results.append(clean_res)

        pct_fin = int(((i + 1) / total) * 100)
        if progress_callback:
            progress_callback(pct_fin,
                              t("progress.done", name=_display_label(key)))

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


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

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
    tag = {
        "pos": t("status.pos"),
        "neg": t("status.neg"),
        "warn": t("status.warn"),
    }.get(kind, t("status.info"))
    _render(
        f'<div class="stat {kind}"><span class="stat-tag">{tag}</span>'
        f'<span class="stat-txt">{_esc(text)}</span></div>'
    )


def context_ribbon():
    cfg = _cfg()
    fin = cfg["end_date"] or datetime.date.today()
    cells = [
        (t("ribbon.instrument"), cfg["ticker"]),
        (t("ribbon.period"), f"{cfg['start_date']} → {fin}"),
        (t("ribbon.window"), f"{cfg['window_size']} {t('ribbon.days')}"),
        (t("ribbon.refit"), f"{cfg['refit_freq']} {t('ribbon.days')}"),
        (t("ribbon.alpha_var"), f"{float(cfg['alpha_var']) * 100:.2f}%"),
        (t("ribbon.alpha_es"), f"{float(cfg['alpha_es']) * 100:.2f}%"),
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
    if key == "basel_zone_actu":
        s = str(v)
        if s.lower() in ("nan", "none"):
            return "—"
        return t(f"zone.{s.upper()}")
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
    head = "".join(f'<th class="num">{_esc(_display_label(m))}</th>' for m in models)
    code_th = "" if compact else f"<th>{_esc(t('ui.code'))}</th>"
    rows = ""
    for key in df_summary.index:
        label = _metric_label(key)
        code_td = "" if compact else f'<td class="code">{_esc(key)}</td>'
        cells = ""
        for m in models:
            raw = df_summary.loc[key, m]
            cells += f'<td class="num {_cell_tone(key, raw)}">{_esc(_fmt(key, raw))}</td>'
        rows += f'<tr><td class="lbl">{_esc(label)}</td>{code_td}{cells}</tr>'
    _render(
        f'<div class="tbl-wrap"><table class="fin"><thead><tr>'
        f'<th>{_esc(t("ui.metric"))}</th>'
        f'{code_th}{head}</tr></thead><tbody>{rows}</tbody></table></div>'
    )


def render_kv_table(rows):
    body = "".join(
        f'<tr><td class="lbl">{_esc(k)}</td><td class="num">{_esc(v)}</td></tr>'
        for k, v in rows
    )
    _render(
        f'<div class="tbl-wrap"><table class="fin"><thead><tr>'
        f'<th>{_esc(t("ui.parameter"))}</th>'
        f'<th class="num">{_esc(t("ui.value"))}</th></tr></thead><tbody>{body}</tbody></table></div>'
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
        x=returns_s.index, y=returns_s.values, mode="lines",
        name=t("chart.returns"),
        line=dict(color=C["ret"], width=0.9),
    ))
    fig.add_trace(go.Scatter(
        x=var_plot.index, y=var_plot.values, mode="lines",
        name=t("chart.var", conf=conf_label),
        line=dict(color=C["navy"], width=1.6, shape="hv"),
    ))
    if "es_frtb_series" in raw_res:
        es_plot = _neg(raw_res["es_frtb_series"])
        fig.add_trace(go.Scatter(
            x=es_plot.index, y=es_plot.values, mode="lines",
            name=t("chart.es"),
            line=dict(color=C["warn"], width=1.2, dash="dot", shape="hv"),
        ))
    exc_mask = returns_s < var_plot
    fig.add_trace(go.Scatter(
        x=returns_s[exc_mask].index, y=returns_s[exc_mask].values,
        mode="markers", name=t("chart.exceptions"),
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
        x=r.index, y=r.values, mode="lines",
        name=t("chart.returns"),
        line=dict(color=C["ret_soft"], width=0.8),
    ))
    for i, (key, res) in enumerate(dict_outputs.items()):
        if "var_series" not in res:
            continue
        v = _neg(res["var_series"])
        fig.add_trace(go.Scatter(
            x=v.index, y=v.values, mode="lines",
            name=t("chart.var_model", conf=conf_label, name=_display_label(key)),
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


# ---------------------------------------------------------------------------
# Composite sections
# ---------------------------------------------------------------------------

def afficher_synthese_comparative(df_summary):
    section("05", t("sec.synthesis"), t("sec.synthesis.desc"))
    try:
        cfg = _cfg()
        model_keys = df_summary.columns.tolist()

        kupiec = {m: float(df_summary.loc["kupiec_pval", m]) for m in model_keys}
        best_kupiec = max(kupiec, key=kupiec.get)

        cc = {m: float(df_summary.loc["christof_cc_pval", m]) for m in model_keys}
        best_cc = max(cc, key=cc.get)

        dq = {m: float(df_summary.loc["dq_pval", m]) for m in model_keys}
        best_dq = max(dq, key=dq.get)

        exc = {m: float(df_summary.loc["exceptions", m]) for m in model_keys}
        n_obs = float(df_summary.loc["obs_tested", model_keys[0]])
        target = n_obs * cfg["alpha_var"]
        best_exc = min(exc, key=lambda m: abs(exc[m] - target))

        rows = [
            (t("synth.var_accuracy"), "Kupiec POF",
             _display_label(best_kupiec), f"p = {kupiec[best_kupiec]:.4f}"),
            (t("synth.exc_indep"), "Christoffersen CC",
             _display_label(best_cc), f"p = {cc[best_cc]:.4f}"),
            (t("synth.dynamic_quality"), "Dynamic Quantile",
             _display_label(best_dq), f"p = {dq[best_dq]:.4f}"),
            (t("synth.exc_calibration"), t("synth.deviation_target"),
             _display_label(best_exc),
             t("synth.obs_target",
               n=int(exc[best_exc]), t=f"{target:.0f}")),
        ]
        body = "".join(
            f'<tr><td class="lbl">{_esc(a)}</td><td class="code">{_esc(b)}</td>'
            f'<td class="lbl strong">{_esc(c)}</td><td class="num">{_esc(d)}</td></tr>'
            for a, b, c, d in rows
        )
        _render(
            f'<div class="tbl-wrap"><table class="fin"><thead><tr>'
            f'<th>{_esc(t("ui.criterion"))}</th>'
            f'<th>{_esc(t("ui.test"))}</th>'
            f'<th>{_esc(t("ui.best_model"))}</th>'
            f'<th class="num">{_esc(t("ui.indicator"))}</th>'
            f'</tr></thead><tbody>{body}</tbody></table></div>'
        )
    except Exception:
        status_line("", t("synth.unavailable"))


def gestionnaire_modeles_custom(show_checkboxes=True):
    if "custom_models" not in st.session_state:
        st.session_state["custom_models"] = {}

    with st.container(border=True):
        panel_label(t("ui.import_model"))
        custom_name = st.text_input(t("ui.model_name"), placeholder=t("ui.model_name.ph"),
                                    key="custom_model_name")

        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            panel_label(t("ui.var_series"))
            file_var = st.file_uploader("VaR", type=["csv", "xlsx", "parquet"], key="add_var",
                                        label_visibility="collapsed")
        with c2:
            panel_label(t("ui.es_series"))
            file_es = st.file_uploader("ES", type=["csv", "xlsx", "parquet"], key="add_es",
                                       label_visibility="collapsed")
        with c3:
            panel_label(t("ui.vol_series"))
            file_vol = st.file_uploader("Vol", type=["csv", "xlsx", "parquet"], key="add_vol",
                                        label_visibility="collapsed")

        st.caption(t("ui.upload_caption"))

        if st.button(t("ui.save_model"), type="primary"):
            clean_name = custom_name.strip()
            if not clean_name:
                status_line("neg", t("custom.name_required"))
            elif clean_name in st.session_state["custom_models"]:
                status_line("neg", t("custom.name_exists"))
            elif not (file_var and file_es and file_vol):
                status_line("neg", t("custom.all_required"))
            else:
                try:
                    var_series = parse_uploaded_file(file_var, t("custom.var_kind"))
                    es_series = parse_uploaded_file(file_es, t("custom.es_kind"))
                    vol_series = parse_uploaded_file(file_vol, t("custom.vol_kind"))
                except ValueError as e:
                    status_line("neg", str(e))
                else:
                    st.session_state["custom_models"][clean_name] = {
                        "var": var_series,
                        "es": es_series,
                        "vol": vol_series,
                    }
                    st.toast(t("custom.saved", name=clean_name))
                    st.rerun()

    if st.session_state["custom_models"]:
        panel_label(t("ui.saved_models.select") if show_checkboxes else t("ui.saved_models"))
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
                        f'<div class="card-desc">{_esc(t("ui.external_model_desc"))}</div>'
                    )

                if show_checkboxes:
                    with col_chk:
                        cols_custom_select[c_name] = st.checkbox(
                            t("ui.include_in_audit"), value=True, key=f"chk_{c_name}")

                with col_del:
                    if st.button(t("ui.remove"), key=f"del_{c_name}", use_container_width=True):
                        del st.session_state["custom_models"][c_name]
                        st.rerun()
        return cols_custom_select
    return {}


def page_audit_comparatif():
    cfg = _cfg()
    page_header(t("header.validation.kicker"), t("header.comparative.title"),
                t("header.comparative.subtitle"))
    context_ribbon()

    section("01", t("sec.reference_models"), t("sec.reference_models.desc"))
    defs = [code for code in STANDARD_MODEL_CODES]
    selected_standards = []
    for col, code in zip(st.columns(3, gap="small"), defs):
        with col:
            with st.container(border=True):
                _render(
                    f'<div class="card-title">{_esc(t(f"model.{code}"))}</div>'
                    f'<div class="card-desc">{_esc(t(f"model.{code}.desc"))}</div>'
                )
                if st.checkbox(t("ui.include_in_audit"), value=True, key=f"model_{code}"):
                    selected_standards.append(code)

    section("02", t("sec.custom_models"), t("sec.custom_models.desc"))
    custom_selections = gestionnaire_modeles_custom()
    selected_customs = {
        name: st.session_state["custom_models"][name]
        for name, is_sel in custom_selections.items() if is_sel
    }

    section("03", t("sec.execution"))
    if st.button(t("ui.run_comparative"), type="primary"):
        if not selected_standards and not selected_customs:
            status_line("warn", t("err.select_one_model"))
        else:
            ok, msg = _check_period(cfg)
            if not ok:
                status_line("neg", msg)
            else:
                cb, clear = _make_progress()
                try:
                    df_summary, dict_outputs = backtester(selected_standards, selected_customs,
                                                         progress_callback=cb)
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
        section("04", t("sec.results"), f"{res['ticker']} · {t('ui.metric')}")
        render_metrics_table(res["df"])
        st.caption(t("ui.colors_caption"))
        afficher_synthese_comparative(res["df"])
        fig = build_overlay_chart(res["out"], res["conf"])
        if fig is not None:
            section("06", t("sec.compared_var"),
                    t("sec.compared_var.desc", conf=res["conf"]))
            st.plotly_chart(fig, use_container_width=True, theme=None)


def page_audit_unitaire():
    cfg = _cfg()
    page_header(t("header.validation.kicker"), t("header.unitaire.title"),
                t("header.unitaire.subtitle"))
    context_ribbon()

    section("01", t("sec.model_selection"))
    options_map = {t(f"model.{code}"): code for code in STANDARD_MODEL_CODES}
    custom_models = st.session_state.get("custom_models", {})
    for c_name in custom_models.keys():
        options_map[c_name] = "custom"

    modele_choisi_label = st.radio(
        t("ui.select_model"),
        options=list(options_map.keys()),
        horizontal=True,
        label_visibility="collapsed",
    )
    modele_code = options_map[modele_choisi_label]

    section("02", t("sec.custom_models"), t("sec.custom_models.desc"))
    gestionnaire_modeles_custom(show_checkboxes=False)

    section("03", t("sec.execution"))
    if st.button(t("ui.run_unit"), type="primary"):
        ok, msg = _check_period(cfg)
        if not ok:
            status_line("neg", msg)
        else:
            cb, clear = _make_progress()
            selected_stds = [modele_code] if modele_code != "custom" else []
            selected_cust = (
                {modele_choisi_label: custom_models[modele_choisi_label]}
                if modele_code == "custom" else {}
            )
            try:
                df_summary, dict_outputs = backtester(selected_stds, selected_cust,
                                                     progress_callback=cb)
            except ValueError as e:
                clear()
                status_line("neg", str(e))
            else:
                clear()
                st.session_state["last_unit"] = {
                    "df": df_summary,
                    "out": dict_outputs,
                    "code": modele_code if modele_code != "custom" else modele_choisi_label,
                    "conf": _conf_label(cfg),
                    "ticker": cfg["ticker"],
                }

    res = st.session_state.get("last_unit")
    if not res:
        return

    model_key = res["code"]
    label = _display_label(model_key)
    df_summary = res["df"]
    section("04", t("sec.results"), f"{label} · {res['ticker']}")

    zone = str(df_summary.loc["basel_zone_actu", model_key])
    n_exc = int(float(df_summary.loc["exceptions", model_key]))
    rate = float(df_summary.loc["exc_rate", model_key])
    kupiec_p = float(df_summary.loc["kupiec_pval", model_key])
    try:
        n_obs = float(df_summary.loc["obs_tested", model_key])
        target_txt = t("kpi.target", n=f"{n_obs * cfg['alpha_var']:.0f}")
    except Exception:
        target_txt = "—"

    zone_tone = {"VERTE": "pos", "JAUNE": "warn", "ROUGE": "neg"}.get(zone.upper(), "")
    zone_display = t(f"zone.{zone.upper()}")

    kpi_strip([
        (t("kpi.basel"), zone_display, t("kpi.basel.sub"), zone_tone),
        (t("kpi.var_exc"), f"{n_exc}", target_txt, ""),
        (t("kpi.exc_rate"), f"{rate:.2f}%", f"VaR {res['conf']}", ""),
        (t("kpi.kupiec"), f"{kupiec_p:.4f}",
         t("kpi.h0_not_rejected") if kupiec_p > 0.05 else t("kpi.h0_rejected"),
         "pos" if kupiec_p > 0.05 else "neg"),
    ])

    panel_label(t("kpi.varcaption", conf=res["conf"]))
    fig = build_backtest_chart(res["out"].get(model_key, {}), res["conf"], height=440)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, theme=None)
    else:
        status_line("", t("chart.unavailable_series"))

    section("05", t("sec.metrics_breakdown"), t("sec.metrics_breakdown.desc"))
    half = (len(df_summary.index) + 1) // 2
    col_l, col_r = st.columns(2, gap="medium")
    with col_l:
        render_metrics_table(df_summary.iloc[:half], compact=True)
    with col_r:
        render_metrics_table(df_summary.iloc[half:], compact=True)


def page_config():
    page_header(t("header.config.kicker"), t("header.config.title"),
                t("header.config.subtitle"))
    cfg = _cfg()

    col_a, col_b = st.columns(2, gap="medium")

    with col_a:
        section("01", t("sec.analysis_period"))
        raw_start = st.text_input(t("ui.start_date"), str(cfg["start_date"]))
        raw_end = st.text_input(t("ui.end_date"), str(cfg["end_date"]))

        try:
            new_start = datetime.date.fromisoformat(raw_start.strip())
            new_end = (None if raw_end.strip().lower() == "none"
                       else datetime.date.fromisoformat(raw_end.strip()))
        except ValueError:
            status_line("neg", t("err.date_format"))
        else:
            end_eff = new_end or datetime.date.today()
            jours_bourse = (end_eff - new_start).days * 252 / 365
            if end_eff <= new_start:
                status_line("neg", t("err.date_order"))
            elif jours_bourse <= cfg["window_size"]:
                status_line("neg", t("err.period_too_short",
                                     n=int(jours_bourse), w=cfg["window_size"]))
            else:
                cfg["start_date"] = new_start
                cfg["end_date"] = new_end
                status_line("pos", t("ui.valid_period"))

    with col_b:
        section("02", t("sec.instrument"))
        raw = st.text_input(t("ui.ticker"), cfg["ticker"])
        symbol = raw.strip().upper()
        if not symbol:
            status_line("warn", t("err.ticker_required"))
        elif is_valid_ticker(symbol):
            cfg["ticker"] = symbol
            status_line("pos", t("ok.ticker_valid", symbol=symbol))
        else:
            status_line("neg", t("err.ticker_not_found", symbol=symbol))

    section("03", t("sec.backtest_params"), t("sec.backtest_params.desc"))
    render_kv_table([
        (t("cfg.window_size"), cfg["window_size"]),
        (t("cfg.refit_freq"), cfg["refit_freq"]),
        (t("cfg.alpha_var"), cfg["alpha_var"]),
        (t("cfg.alpha_es"), cfg["alpha_es"]),
        (t("cfg.lags_dq"), CONFIG.get("lags_dq", "—")),
    ])


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

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

[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stCheckbox"] label p,
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] [data-testid="stToggle"] p {
    color: #C4CFDD !important;
    font-size: 12.5px;
    font-weight: 500;
    letter-spacing: 0.02em;
}

[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] [data-testid="stCheckbox"] label,
[data-testid="stSidebar"] [data-testid="stToggle"] label {
    display: flex;
    align-items: center;
    gap: 10px;
}

[data-testid="stSidebar"] [data-testid="stCheckbox"] label p,
[data-testid="stSidebar"] [data-testid="stToggle"] label p {
    margin: 0;
    line-height: 1;
}

[data-testid="stSidebar"] [data-baseweb="checkbox"] > div:first-child {
    margin-top: 0;
    align-self: center;
}
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


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

def afficher_interface():
    st.set_page_config(page_title="Audit Operator", layout="wide")

    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "light"
    if "lang" not in st.session_state:
        st.session_state["lang"] = "fr"

    st.markdown(CSS.replace("/*ROOT*/", apply_theme()), unsafe_allow_html=True)

    cfg = _cfg()

    st.sidebar.markdown(_h(f"""
    <div>
        <div class="sb-brand">AUDIT OPERATOR</div>
        <div class="sb-sub">{t("sb.tagline")}</div>
        <div class="sb-author">Adam Khazani Ahsaini</div>
    </div>
    """), unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f'<div class="sb-label">{_esc(t("sb.navigation"))}</div>',
                        unsafe_allow_html=True)

    menu_labels = {
        "comparative": t("sb.menu.comparative"),
        "unitaire": t("sb.menu.unitaire"),
        "config": t("sb.menu.config"),
    }
    menu = st.sidebar.radio(
        t("sb.navigation"),
        options=list(menu_labels.keys()),
        format_func=lambda k: menu_labels[k],
        label_visibility="collapsed",
        key="menu_choice",
    )

    st.sidebar.markdown("---")

    is_dark = st.sidebar.toggle(t("sb.dark"),
                                value=(st.session_state["theme_mode"] == "dark"))
    if is_dark and st.session_state["theme_mode"] != "dark":
        st.session_state["theme_mode"] = "dark"
        st.rerun()
    elif not is_dark and st.session_state["theme_mode"] != "light":
        st.session_state["theme_mode"] = "light"
        st.rerun()

    # --- Language selector (FR by default, EN / ES / ZH / DE available) ---
    current_lang = st.session_state.get("lang", "fr")
    if current_lang not in LANG_OPTIONS:
        current_lang = "fr"
        st.session_state["lang"] = "fr"

    selected_lang = st.sidebar.selectbox(
        t("sb.language"),
        options=list(LANG_OPTIONS.keys()),
        format_func=lambda k: LANG_OPTIONS[k],
        index=list(LANG_OPTIONS.keys()).index(current_lang),
        key="lang_select",
    )
    if selected_lang != current_lang:
        st.session_state["lang"] = selected_lang
        st.rerun()

    st.sidebar.markdown("---")

    date_fin = cfg["end_date"] or datetime.date.today()
    st.sidebar.markdown(_h(f"""
    <div class="sb-label">{_esc(t("sb.active_market"))}</div>
    <div class="sb-tick">{_esc(cfg['ticker'])}</div>
    <div class="sb-per">{_esc(cfg['start_date'])} → {_esc(date_fin)}</div>
    """), unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f'<div class="sb-foot">{t("sb.footer")}</div>',
                        unsafe_allow_html=True)

    if menu == "comparative":
        page_audit_comparatif()
    elif menu == "unitaire":
        page_audit_unitaire()
    elif menu == "config":
        page_config()


if __name__ == "__main__":
    if not st.runtime.exists():
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
    else:
        afficher_interface()