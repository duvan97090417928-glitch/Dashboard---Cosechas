import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Cosechas", layout="wide")

REQUIRED_COLS = [
    "FechaCierre", "FechaDesembolso", "NumeroCredito",
    "MontoDesembolso", "SaldoColocaciones", "DiasMora",
]

FILTER_CANDIDATES = [
    "Clasificacion", "modalidad", "plazo", "Tasa de interés",
    "seccion", "canal", "medio de pago", "Linea de credito",
    "Destinacion", "Variable 9", "Variable 10",
]

MAX_COHORTS_ANNOTATE = 24
MAX_CELLS_ANNOTATE = 700


# -----------------------------
# HELPERS
# -----------------------------
def validate_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en el archivo: {missing}")


def safe_to_category(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def normalize_filters(selected_filters: dict) -> tuple:
    items = []
    for k in sorted(selected_filters.keys()):
        vals = selected_filters[k]
        if not vals:
            continue
        vals = tuple(sorted(map(str, vals)))
        items.append((k, vals))
    return tuple(items)


def apply_filters_fast(df: pd.DataFrame, selected_filters: dict) -> pd.DataFrame:
    out = df
    for col, vals in selected_filters.items():
        if vals:
            out = out[out[col].isin(vals)]
    return out


def build_excel_bytes(agg, mora_mat, des_mat, vint_pct) -> io.BytesIO:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        agg.to_excel(writer, sheet_name="Base_Agg", index=False)
        mora_mat.to_excel(writer, sheet_name="Mora")
        des_mat.to_excel(writer, sheet_name="Desembolso")
        vint_pct.to_excel(writer, sheet_name="Vintage_%")
    out.seek(0)
    return out


# -----------------------------
# LOAD (CACHED)
# -----------------------------
@st.cache_data(show_spinner="📥 Leyendo Parquet...")
def load_parquet(uploaded_file) -> pd.DataFrame:
    return pd.read_parquet(uploaded_file)


@st.cache_data(show_spinner="⚙️ Preparando datos (1 sola vez)...")
def prepare_data(df_raw: pd.DataFrame):
    df = df_raw.copy()
    validate_columns(df)

    # Fechas
    if df["FechaCierre"].dtype == "object":
        df["FechaCierre"] = pd.to_datetime(df["FechaCierre"], dayfirst=True, errors="coerce")
    else:
        df["FechaCierre"] = pd.to_datetime(df["FechaCierre"], errors="coerce")

    if df["FechaDesembolso"].dtype == "object":
        df["FechaDesembolso"] = pd.to_datetime(df["FechaDesembolso"], dayfirst=True, errors="coerce")
    else:
        df["FechaDesembolso"] = pd.to_datetime(df["FechaDesembolso"], errors="coerce")

    df = df.dropna(subset=["FechaCierre", "FechaDesembolso"])

    # Tipos base
    df["NumeroCredito"] = df["NumeroCredito"].astype(str)
    df["DiasMora"] = pd.to_numeric(df["DiasMora"], errors="coerce").fillna(0).astype("int16")
    df["SaldoColocaciones"] = pd.to_numeric(df["SaldoColocaciones"], errors="coerce").fillna(0.0).astype("float64")
    df["MontoDesembolso"] = pd.to_numeric(df["MontoDesembolso"], errors="coerce").fillna(0.0).astype("float64")

    # Cohorte (YYYY-MM)
    df["Cosecha"] = df["FechaDesembolso"].dt.to_period("M").astype(str)

    # MOB
    mob = (
        (df["FechaCierre"].dt.year - df["FechaDesembolso"].dt.year) * 12
        + (df["FechaCierre"].dt.month - df["FechaDesembolso"].dt.month)
    )
    df["MOB"] = pd.to_numeric(mob, errors="coerce").fillna(-999).astype("int16")

    # Filtros existentes
    available_filters = [c for c in FILTER_CANDIDATES if c in df.columns]
    df = safe_to_category(df, available_filters)

    panel_cols = REQUIRED_COLS + ["Cosecha", "MOB"] + available_filters
    panel_df = df[panel_cols].copy()

    # Crédito-level: 1 fila por crédito (denominador fijo por cohorte)
    credit_cols = ["NumeroCredito", "Cosecha", "MontoDesembolso"] + available_filters
    credit_df = (
        panel_df.sort_values(["NumeroCredito", "FechaDesembolso"])
                .drop_duplicates(subset=["NumeroCredito"], keep="first")[credit_cols]
                .copy()
    )

    # Meses disponibles (YYYY-MM)
    min_month = panel_df["FechaDesembolso"].min().to_period("M")
    max_month = panel_df["FechaDesembolso"].max().to_period("M")
    months = pd.period_range(min_month, max_month, freq="M").astype(str).tolist()

    # Opciones de filtros
    filter_options = {}
    for c in available_filters:
        vals = panel_df[c].dropna()
        if str(vals.dtype) == "category":
            opts = list(vals.cat.categories)
        else:
            opts = sorted(vals.unique().tolist(), key=lambda x: str(x))
        filter_options[c] = opts

    return panel_df, credit_df, available_filters, filter_options, months


# -----------------------------
# COMPUTE (CACHED)
# -----------------------------
@st.cache_data(show_spinner="📊 Calculando cosechas...")
def compute_vintage(
    panel_df: pd.DataFrame,
    credit_df: pd.DataFrame,
    mora_dias: int,
    max_mob: int,
    cierre_str: str | None,
    mes_desde: str,
    mes_hasta: str,
    filt_key: tuple
):
    selected_filters = {k: list(v) for k, v in filt_key}

    d0 = pd.Period(mes_desde, freq="M").to_timestamp(how="start")
    d1 = pd.Period(mes_hasta, freq="M").to_timestamp(how="end")

    p = panel_df[
        (panel_df["FechaDesembolso"] >= d0) &
        (panel_df["FechaDesembolso"] <= d1) &
        (panel_df["MOB"] >= 0) &
        (panel_df["MOB"] <= max_mob)
    ]
    p = apply_filters_fast(p, selected_filters)

    # Cierre exacto opcional
    if cierre_str:
        cierre_dt = pd.to_datetime(cierre_str, errors="coerce")
        if pd.notna(cierre_dt):
            p = p[p["FechaCierre"] == cierre_dt]

    if p.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    all_mobs = list(range(0, max_mob + 1))

    # ✅ Mora con ceros: MoraX=Saldo si DiasMora>mora_dias, si no 0
    p = p.copy()
    p["MoraX"] = np.where(p["DiasMora"] > mora_dias, p["SaldoColocaciones"], 0.0)

    mora_full = (
        p.groupby(["Cosecha", "MOB"], as_index=False)["MoraX"]
         .sum()
         .rename(columns={"MoraX": "Mora"})
    )

    # Denominador fijo por cohorte (sum MontoDesembolso, 1 fila por crédito)
    c = apply_filters_fast(credit_df, selected_filters)
    c = c[(c["Cosecha"] >= mes_desde) & (c["Cosecha"] <= mes_hasta)]

    desembolso = (
        c.groupby("Cosecha", as_index=False)["MontoDesembolso"]
         .sum()
         .rename(columns={"MontoDesembolso": "Desembolso"})
    )

    agg = mora_full.merge(desembolso, on="Cosecha", how="left")
    agg["Vintage"] = np.where(agg["Desembolso"] > 0, agg["Mora"] / agg["Desembolso"], np.nan)

    vint_mat = agg.pivot(index="Cosecha", columns="MOB", values="Vintage").reindex(columns=all_mobs)
    mora_mat = agg.pivot(index="Cosecha", columns="MOB", values="Mora").reindex(columns=all_mobs)

    # ✅ Desembolso triangular: solo donde existe observación (vintage no NaN)
    des_series = desembolso.set_index("Cosecha")["Desembolso"]

    des_const = pd.DataFrame(index=vint_mat.index, columns=vint_mat.columns, dtype="float64")
    for m in all_mobs:
        des_const[m] = des_series.reindex(des_const.index)

    mask_obs = vint_mat.notna()
    des_mat = des_const.where(mask_obs, np.nan)

    vint_pct = (vint_mat * 100).round(2)

    return agg, mora_mat, des_mat, vint_pct


# -----------------------------
# PLOTS
# -----------------------------
def render_heatmap_fast(vint_pct: pd.DataFrame, title: str, force_values: bool):
    if vint_pct.empty:
        st.warning("No hay datos con los filtros seleccionados.")
        return

    mat = vint_pct.to_numpy()
    ylabels = vint_pct.index.tolist()
    xlabels = vint_pct.columns.tolist()

    fig, ax = plt.subplots(figsize=(15, 7))
    from matplotlib.colors import PowerNorm
    import matplotlib.cm as cm

    cmap = cm.get_cmap("RdYlGn_r")

    vmin = 0
    vmax = np.nanmax(mat) if np.isfinite(np.nanmax(mat)) else 1

    # gamma < 1 => menos “verde dominante”, más contraste en medios/altos
    norm = PowerNorm(gamma=0.35, vmin=vmin, vmax=vmax)

    im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)

    ax.set_title(title)
    ax.set_ylabel("Cosecha (YYYY-MM)")
    ax.set_xlabel("MOB (meses)")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)

    auto_annotate = (len(ylabels) <= MAX_COHORTS_ANNOTATE) and (mat.size <= MAX_CELLS_ANNOTATE)
    do_annotate = force_values or auto_annotate

    if do_annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    st.pyplot(fig)


def render_lines(vint_pct: pd.DataFrame, cosechas_sel: list[str], max_mob: int):
    if vint_pct.empty:
        return

    v = vint_pct.copy()
    if cosechas_sel:
        v = v.loc[v.index.isin(cosechas_sel)]

    cols = list(range(0, max_mob + 1))
    v = v[cols].T

    fig, ax = plt.subplots(figsize=(12, 5))
    v.plot(ax=ax)
    ax.set_title("Vintage por MOB")
    ax.set_xlabel("MOB (meses)")
    ax.set_ylabel("% Vintage (Mora/Desembolso)")
    ax.set_xlim(0, max_mob)
    ax.set_xticks(np.arange(0, max_mob + 1, 1))
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


# -----------------------------
# UI
# -----------------------------
st.title("📊 Dashboard de Cosechas (Vintage)")

with st.sidebar:
    st.header("📥 Cargar datos")
    uploaded = st.file_uploader("Sube data_blog.parquet", type=["parquet"])

if uploaded is None:
    st.info("Sube el archivo .parquet para empezar.")
    st.stop()

progress = st.progress(0, text="Iniciando...")
try:
    progress.progress(20, text="Leyendo Parquet...")
    df_raw = load_parquet(uploaded)

    progress.progress(70, text="Preparando columnas...")
    panel_df, credit_df, available_filters, filter_options, months = prepare_data(df_raw)

    progress.progress(100, text="✅ Listo")
    progress.empty()
except Exception as e:
    progress.empty()
    st.error(f"Error leyendo/preparando el archivo: {e}")
    st.stop()

# Estado para no recalcular pesado cada click
if "computed" not in st.session_state:
    st.session_state.computed = False
    st.session_state.agg = None
    st.session_state.mora_mat = None
    st.session_state.des_mat = None
    st.session_state.vint_pct = None
    st.session_state.last_params = None

# -----------------------------
# FORM: filtros pesados (con botón)
# -----------------------------
with st.sidebar.form("form_filtros"):
    st.header("🎛️ Parámetros (pesados)")
    mora_dias = st.selectbox("Mora (días) >", options=[8, 30, 60, 90], index=1)
    max_mob = st.slider("Máximo MOB (meses)", min_value=3, max_value=36, value=12, step=1)
    cierre_str = st.text_input("FechaCierre exacta (opcional, YYYY-MM-DD)", value="").strip()

    st.divider()
    st.header("📅 Rango de cosecha (por mes)")

    default_desde = "2023-01" if "2023-01" in months else months[max(0, len(months) - 24)]
    default_hasta = "2023-12" if "2023-12" in months else months[-1]

    mes_desde = st.selectbox("Cosecha desde (YYYY-MM)", months, index=months.index(default_desde))
    mes_hasta = st.selectbox("Cosecha hasta (YYYY-MM)", months, index=months.index(default_hasta))

    st.divider()
    st.header("🧩 Filtros dinámicos (pesados)")
    selected_filters = {}
    for col in available_filters:
        opts = filter_options.get(col, [])
        chosen = st.multiselect(col, options=opts, default=[])
        if chosen:
            selected_filters[col] = chosen

    aplicar = st.form_submit_button("✅ Aplicar filtros")

# Validación simple
if mes_desde > mes_hasta:
    st.error("El mes 'desde' no puede ser mayor que el mes 'hasta'.")
    st.stop()

# Si aplicó, recalcula y guarda en session_state
if aplicar:
    filt_key = normalize_filters(selected_filters)

    params = (mora_dias, max_mob, cierre_str if cierre_str else None, mes_desde, mes_hasta, filt_key)

    agg, mora_mat, des_mat, vint_pct = compute_vintage(
        panel_df=panel_df,
        credit_df=credit_df,
        mora_dias=mora_dias,
        max_mob=max_mob,
        cierre_str=cierre_str if cierre_str else None,
        mes_desde=mes_desde,
        mes_hasta=mes_hasta,
        filt_key=filt_key
    )

    st.session_state.agg = agg
    st.session_state.mora_mat = mora_mat
    st.session_state.des_mat = des_mat
    st.session_state.vint_pct = vint_pct
    st.session_state.computed = True
    st.session_state.last_params = params

# Si aún no ha aplicado, pide aplicar una vez
if not st.session_state.computed:
    st.info("Configura filtros en la barra izquierda y presiona **Aplicar filtros** una vez. Luego, el selector de líneas será automático.")
    st.stop()

# Recuperar resultados
agg = st.session_state.agg
mora_mat = st.session_state.mora_mat
des_mat = st.session_state.des_mat
vint_pct = st.session_state.vint_pct

# -----------------------------
# CONTROLES LIGEROS (AUTOMÁTICOS)
# -----------------------------
st.subheader("⚙️ Controles rápidos")
cA, cB = st.columns(2)
with cA:
    force_values = st.checkbox("Mostrar valores en heatmap (aplica para heatmap grandes)", value=False)
with cB:
    show_tables = st.checkbox("Mostrar tablas (Mora/Desembolso/Vintage)", value=False)

# -----------------------------
# Resumen
# -----------------------------
st.subheader("✅ Resumen")
c1, c2 = st.columns(2)

# Créditos únicos según filtros + rango de cosecha
selected_filters = {k: list(v) for k, v in normalize_filters(selected_filters)}  # (si ya tienes filt_key, úsalo abajo)
# Mejor: usa filt_key que ya calculas al aplicar:
filt_key = normalize_filters(selected_filters)
sel = {k: list(v) for k, v in filt_key}

c_filtered = apply_filters_fast(credit_df, sel)
c_filtered = c_filtered[(c_filtered["Cosecha"] >= mes_desde) & (c_filtered["Cosecha"] <= mes_hasta)]

creditos_unicos_filtrados = c_filtered["NumeroCredito"].nunique()
c1.metric("Créditos únicos (filtrado)", f"{creditos_unicos_filtrados:,}".replace(",", "."))
c2.metric("Cosechas resultantes", "0" if vint_pct.empty else f"{vint_pct.shape[0]:,}".replace(",", "."))

# -----------------------------
# Heatmap
# -----------------------------
st.subheader("🔥 Heatmap (Vintage %)")
render_heatmap_fast(
    vint_pct,
    title=f"Vintage (%) | Mora>{mora_dias} | MOB<= {max_mob} | {mes_desde} a {mes_hasta}",
    force_values=force_values
)

# -----------------------------
# Líneas (AUTO, sin aplicar)
# -----------------------------
st.subheader("📈 Gráfica de Líneas (selección de cosechas)")

if not vint_pct.empty:
    cosechas_disponibles = sorted(vint_pct.index.tolist())
    default_sel = cosechas_disponibles[:3] if len(cosechas_disponibles) >= 3 else cosechas_disponibles

    cosechas_sel = st.multiselect(
        "Selecciona cosechas (YYYY-MM)",
        options=cosechas_disponibles,
        default=st.session_state.get("line_cohorts", default_sel),
        key="line_cohorts"
    )

    render_lines(vint_pct, cosechas_sel, max_mob)

# -----------------------------
# Tablas (opcional)
# -----------------------------
if show_tables:
    st.subheader("📋 Tablas")
    st.write("**Mora**")
    st.dataframe(mora_mat.fillna(0))
    st.write("**Desembolso**")
    st.dataframe(des_mat)
    st.write("**Vintage %**")
    st.dataframe(vint_pct)

# -----------------------------
# Export
# -----------------------------
st.subheader("⬇️ Descargar salida")
if not vint_pct.empty:
    excel_bytes = build_excel_bytes(agg, mora_mat, des_mat, vint_pct)
    st.download_button(
        label="Descargar Excel (Base_Agg + matrices)",
        data=excel_bytes,
        file_name=f"salida_cosechas_{mes_desde}_a_{mes_hasta}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("No hay resultados para exportar con esos filtros.")