import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def build_vintage_monto(
    path_xlsx: str,
    sheet: str = "Data",
    modalidad: str = "1 - Consumo",
    cosecha_desde: str = "2023-01-01",
    cosecha_hasta: str | None = "2023-12-31",
    cierre: str | None = None,      # Ej: "2024-03-01" o None
    mora_dias: int = 30,
    max_mob: int = 12
):
    """
    Vintage por MONTO (denominador fijo por cohorte):
      - Numerador (por MOB): sum(SaldoColocaciones) donde DiasMora > mora_dias
      - Denominador (por cohorte): sum(MontoDesembolso) una sola vez por crédito (no por MOB)
      - Cohorte: mes de FechaDesembolso
      - MOB: meses entre FechaDesembolso y FechaCierre (entero)
    """

    # 1) Leer datos
    df = pd.read_excel(path_xlsx, sheet_name=sheet)

    # 2) Convertir fechas
    df["FechaCierre"] = pd.to_datetime(df["FechaCierre"], dayfirst=True, errors="coerce")
    df["FechaDesembolso"] = pd.to_datetime(df["FechaDesembolso"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["FechaCierre", "FechaDesembolso"])

    # 3) Filtros base
    df = df[df["modalidad"] == modalidad].copy()

    d0 = pd.to_datetime(cosecha_desde)
    df = df[df["FechaDesembolso"] >= d0]

    if cosecha_hasta is not None:
        d1 = pd.to_datetime(cosecha_hasta)
        df = df[df["FechaDesembolso"] <= d1]

    # 4) (Opcional) filtrar por un cierre específico
    if cierre is not None:
        cierre_dt = pd.to_datetime(cierre)
        df = df[df["FechaCierre"] == cierre_dt]

    # 5) Cohorte y MOB
    df["Cosecha"] = df["FechaDesembolso"].dt.to_period("M")
    df["MOB"] = (
        (df["FechaCierre"].dt.year - df["FechaDesembolso"].dt.year) * 12
        + (df["FechaCierre"].dt.month - df["FechaDesembolso"].dt.month)
    )

    # Solo MOB que quieres graficar
    df = df[df["MOB"].between(0, max_mob)]

    # 6) Numerador mora (saldo en mora > X)
    df["MoraX"] = np.where(df["DiasMora"] > mora_dias, df["SaldoColocaciones"], 0)

    # ==========================
    # ✅ ÚLTIMA CORRECCIÓN:
    # Denominador fijo por cohorte (NO por MOB)
    # Tomamos 1 fila por crédito para no duplicar MontoDesembolso
    # ==========================
    base_credito = (
        df.sort_values(["NumeroCredito", "FechaDesembolso"])
          .drop_duplicates(subset=["NumeroCredito"], keep="first")
          .loc[:, ["NumeroCredito", "Cosecha", "MontoDesembolso"]]
    )

    desembolso_cohorte = (
        base_credito.groupby("Cosecha", as_index=False)
                   .agg(Desembolso=("MontoDesembolso", "sum"))
    )

    # 7) Numerador por cohorte y MOB
    mora_agg = (
        df.groupby(["Cosecha", "MOB"], as_index=False)
          .agg(Mora=("MoraX", "sum"))
    )

    # 8) Unir denominador fijo a todos los MOB
    agg = mora_agg.merge(desembolso_cohorte, on="Cosecha", how="left")

    # 9) Vintage por MONTO
    agg["Vintage"] = np.where(agg["Desembolso"] > 0, agg["Mora"] / agg["Desembolso"], np.nan)

    # 10) Matrices (lo que se grafica)
    mora_mat = agg.pivot(index="Cosecha", columns="MOB", values="Mora")
    des_mat = agg.pivot(index="Cosecha", columns="MOB", values="Desembolso")
    vint_mat = agg.pivot(index="Cosecha", columns="MOB", values="Vintage")

    # % para heatmap
    vint_pct = (vint_mat * 100).round(2)

    return df, agg, mora_mat, des_mat, vint_pct


def plot_heatmap(vintage_pct: pd.DataFrame, title: str):
    plt.figure(figsize=(15, 8))
    ax = sns.heatmap(vintage_pct, annot=True, fmt=".2f", linewidths=.5, cbar=False)
    ax.xaxis.tick_top()
    ax.xaxis.label_position = "top"
    ax.set(title=title, ylabel="Cosecha (Mes Desembolso)", xlabel="MOB (meses)")
    plt.show()


def plot_lines(vintage_pct: pd.DataFrame, cosechas=None, max_mob: int = 12, title: str = "Vintage por MOB"):
    v = vintage_pct.copy()
    v.index = v.index.astype(str)

    if cosechas is not None:
        v = v.loc[v.index.isin(cosechas)]

    cols = [c for c in v.columns if c <= max_mob]
    v = v[cols].T

    v.plot(figsize=(12, 6))
    plt.title(title)
    plt.xlabel("MOB (meses)")
    plt.ylabel("% Vintage (Mora/Desembolso)")
    plt.xlim(0, max_mob)
    plt.xticks(np.arange(0, max_mob + 1, 1))
    plt.show()


def export_vintage_excel(
    output_path: str,
    agg: pd.DataFrame,
    mora_mat: pd.DataFrame,
    des_mat: pd.DataFrame,
    vintage_pct: pd.DataFrame
):
    mora_out = mora_mat.copy()
    des_out = des_mat.copy()
    vint_out = vintage_pct.copy()

    mora_out.index = mora_out.index.astype(str)
    des_out.index = des_out.index.astype(str)
    vint_out.index = vint_out.index.astype(str)

    agg_out = agg.copy()
    agg_out["Cosecha"] = agg_out["Cosecha"].astype(str)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        agg_out.to_excel(writer, sheet_name="Base_Agg", index=False)
        mora_out.to_excel(writer, sheet_name="Mora")
        des_out.to_excel(writer, sheet_name="Desembolso")
        vint_out.to_excel(writer, sheet_name="Vintage_%")

    print(f"✅ Excel generado: {output_path}")


if __name__ == "__main__":

    archivo = r"D:/Archivos/Visual Studio Code/SARC/Cosechas_v2/data_blog.xlsx"

    df, agg, mora_mat, des_mat, vintage_pct = build_vintage_monto(
        path_xlsx=archivo,
        sheet="Data",
        modalidad="1 - Consumo",
        cosecha_desde="2023-01-01",
        cosecha_hasta="2023-12-31",
        cierre=None,     # o "2024-03-01"
        mora_dias=30,
        max_mob=12
    )

    plot_heatmap(vintage_pct, "Cosechas 2023 (Mora>30 / MontoDesembolso) %")

    plot_lines(
        vintage_pct,
        cosechas=["2023-01", "2023-06", "2023-12"],
        max_mob=12,
        title="Vintage por MOB (selección cosechas 2023)"
    )

    salida = r"D:/Archivos/Visual Studio Code/SARC/Cosechas_v2/salida_cosechas_2023.xlsx"
    export_vintage_excel(salida, agg, mora_mat, des_mat, vintage_pct)
    