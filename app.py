import io
import numpy as np, pandas as pd, streamlit as st
import matplotlib.pyplot as plt, seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.tsaplots import plot_acf

st.set_page_config(page_title="TP Estadística — Castelar", page_icon="📊", layout="wide")
sns.set_theme(style="whitegrid", font_scale=.85)
DEFAULT_CSV = "01_Buenos_Aires_Castelar.csv"

# ---------------------------------------------------------------- datos
@st.cache_data
def cargar(contenido: bytes):
    df = pd.read_csv(io.BytesIO(contenido), sep=None, engine="python")  # detecta ; o ,
    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    return df

st.sidebar.title("📊 TP Métodos Estadísticos para IA")
sub = st.sidebar.file_uploader("Subí tu CSV (opcional)", type="csv")
try:
    contenido = sub.getvalue() if sub else open(DEFAULT_CSV, "rb").read()
    df = cargar(contenido)
except Exception as e:
    st.error(f"No pude leer el CSV: {e}"); st.stop()

num = df.select_dtypes("number").columns.tolist()
if len(num) < 2:
    st.error("El archivo necesita al menos 2 columnas numéricas."); st.stop()
st.sidebar.caption(f"{'Archivo subido' if sub else 'Usando el CSV de ejemplo'} · {len(df)} filas")

pagina = st.sidebar.radio("Pasos del TP", [
    "1. Los datos", "2. Descripción numérica y gráfica", "3. Regresión simple",
    "4. Análisis de residuos", "5. Test de hipótesis de la pendiente", "6. Regresión múltiple"])

def simple(texto, titulo="💡 En palabras simples"):
    with st.expander(titulo, expanded=True):
        st.markdown(texto)

def idx(nombre, lista, defecto=0):
    return lista.index(nombre) if nombre in lista else defecto

# ---------------------------------------------------------------- 1
if pagina.startswith("1"):
    st.title("1. Los datos")
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas (observaciones)", len(df)); c2.metric("Columnas", df.shape[1])
    c3.metric("Valores faltantes", int(df.isna().sum().sum()))
    st.dataframe(df.head(15))
    st.subheader("Tipo de cada variable")
    tipos = pd.DataFrame({"tipo en pandas": df.dtypes.astype(str),
                          "faltantes": df.isna().sum(),
                          "valores distintos": df.nunique()})
    st.dataframe(tipos)
    if "Fecha" in df and df["Fecha"].notna().all():
        dif = df["Fecha"].diff().dropna()
        st.write(f"**Período:** {df['Fecha'].min().date()} → {df['Fecha'].max().date()} · "
                 f"¿un dato por día sin saltos? **{'Sí' if (dif == pd.Timedelta(days=1)).all() else 'No'}**")
    simple("""
Cada **fila** es un día y cada **columna** una medición de ese día.
Las columnas numéricas son **cuantitativas** (se pueden sumar, promediar y comparar);
`Fecha` es una variable de **tiempo**. Antes de calcular nada conviene revisar que no falten datos
ni haya filas repetidas: si los hubiera, habría que decidir qué hacer con ellos.
""")

# ---------------------------------------------------------------- 2
elif pagina.startswith("2"):
    st.title("2. Descripción numérica y gráfica")
    d = df[num]
    t = pd.DataFrame({"media": d.mean(), "mediana": d.median(), "mín": d.min(), "Q1": d.quantile(.25),
                      "Q3": d.quantile(.75), "máx": d.max(), "desvío": d.std(),
                      "CV %": d.std() / d.mean() * 100, "asimetría": d.skew(), "curtosis": d.kurt()}).T
    st.subheader("Medidas de posición y dispersión")
    st.dataframe(t.round(3))
    simple("""
- **Media / mediana:** el valor "típico". Si son parecidas, la distribución es bastante simétrica.
- **Desvío estándar:** cuánto se aleja un día cualquiera del promedio.
- **CV (desvío / media):** dispersión *relativa*; sirve para comparar variables con distintas unidades.
- **Asimetría:** 0 = simétrica; **curtosis** negativa = distribución "achatada" (a veces con dos grupos).
""")
    v = st.selectbox("Elegí una variable para ver su distribución", num, index=len(num) - 1)
    c1, c2 = st.columns([2, 1])
    f, ax = plt.subplots(figsize=(6, 3.2)); sns.histplot(d[v], kde=True, ax=ax, color="steelblue"); c1.pyplot(f)
    f, ax = plt.subplots(figsize=(3, 3.2)); sns.boxplot(y=d[v], ax=ax, color="lightsteelblue"); c2.pyplot(f)
    p = stats.shapiro(d[v]).pvalue
    st.write(f"**Test de normalidad (Shapiro-Wilk):** p = {p:.2e} → "
             + ("**no** se puede descartar que sea normal." if p > .05 else "**se rechaza** la normalidad (no es una campana perfecta)."))
    if "Fecha" in df:
        st.subheader("Evolución en el tiempo")
        f, ax = plt.subplots(figsize=(11, 2.8)); ax.plot(df["Fecha"], df[v], lw=.8); ax.set_ylabel(v); st.pyplot(f)
        simple("Si la línea sube y baja con ciclos repetidos (por ejemplo verano/invierno) hay **estacionalidad**: "
               "los días cercanos se parecen entre sí, y eso afectará más adelante a la regresión.")
    st.subheader("Correlaciones")
    f, ax = plt.subplots(figsize=(5.5, 4.2)); sns.heatmap(d.corr(), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax); st.pyplot(f)
    simple("La **correlación** va de −1 a +1. Cerca de +1: suben juntas. Cerca de −1: una sube cuando la otra baja. "
           "Cerca de 0: no hay relación *lineal*.")

# ---------------------------------------------------------------- 3
elif pagina.startswith("3"):
    st.title("3. Regresión lineal simple")
    c1, c2 = st.columns(2)
    xn = c1.selectbox("Variable explicativa (X)", num, index=idx("Temperatura_Abrigo_150cm", num))
    yopts = [c for c in num if c != xn]
    yn = c2.selectbox("Variable a explicar (Y)", yopts, index=idx("Radiacion_Global", yopts))
    X = sm.add_constant(df[xn]); y = df[yn]; m = sm.OLS(y, X).fit()
    b0, b1 = m.params
    st.success(f"**Recta ajustada:**  {yn} = {b0:.3f} + {b1:.3f} · {xn}")
    k1, k2, k3 = st.columns(3)
    k1.metric("Pendiente (β₁)", f"{b1:.3f}"); k2.metric("R²", f"{m.rsquared:.3f}"); k3.metric("Correlación r", f"{np.corrcoef(df[xn], y)[0,1]:.3f}")
    f, ax = plt.subplots(figsize=(7, 4))
    sns.regplot(x=xn, y=yn, data=df, scatter_kws=dict(alpha=.35, s=14), line_kws=dict(color="red"), ax=ax); st.pyplot(f)
    simple(f"""
La regresión busca **la recta que mejor atraviesa la nube de puntos** (la que minimiza la suma de los errores al cuadrado).

- **Pendiente = {b1:.3f}:** cuando **{xn}** aumenta 1 unidad, **{yn}** cambia en promedio {b1:.3f} unidades.
- **Intercepto = {b0:.3f}:** valor esperado de Y cuando X = 0 (a veces no tiene sentido práctico, si X=0 queda fuera de lo observado).
- **R² = {m.rsquared:.3f}:** la recta explica el **{m.rsquared*100:.1f} %** de la variabilidad de {yn}. El resto ({(1-m.rsquared)*100:.1f} %) depende de otras cosas.
""")
    st.subheader("🎛️ Probá una predicción")
    xv = st.slider(f"Valor de {xn}", float(df[xn].min()), float(df[xn].max()), float(df[xn].mean()))
    st.write(f"Predicción de **{yn}** = {b0:.3f} + {b1:.3f} × {xv:.2f} = **{b0 + b1*xv:.2f}**")
    SST = ((y - y.mean()) ** 2).sum(); SSE = (m.resid ** 2).sum()
    with st.expander("Cómo se calcula el R²"):
        st.latex(r"R^2=\frac{SSR}{SST}=1-\frac{SSE}{SST}")
        st.write(f"SST (variación total) = {SST:.1f} · SSE (error no explicado) = {SSE:.1f} · SSR (explicada) = {SST-SSE:.1f}  →  R² = {(SST-SSE)/SST:.4f}")
    st.dataframe(m.summary2().tables[1].round(4))

# ---------------------------------------------------------------- 4
elif pagina.startswith("4"):
    st.title("4. Análisis de residuos")
    c1, c2 = st.columns(2)
    xn = c1.selectbox("X", num, index=idx("Temperatura_Abrigo_150cm", num))
    yopts = [c for c in num if c != xn]
    yn = c2.selectbox("Y", yopts, index=idx("Radiacion_Global", yopts))
    X = sm.add_constant(df[xn]); m = sm.OLS(df[yn], X).fit(); r = m.resid; fit = m.fittedvalues
    simple("""
El **residuo** es el error de la recta en cada día: *valor real − valor predicho*.
Si el modelo es bueno, los residuos deben parecer **ruido al azar**: sin forma, de tamaño parejo, sin relación entre días vecinos y con forma de campana.
Si tienen patrones, el modelo se está perdiendo algo.
""")
    s = m.get_influence().resid_studentized_internal
    f, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].scatter(fit, r, alpha=.4, s=12); ax[0, 0].axhline(0, c="r"); ax[0, 0].set(title="Residuos vs. ajustados (¿forma de embudo o curva?)", xlabel="ajustado")
    sm.qqplot(s, line="45", ax=ax[0, 1]); ax[0, 1].set_title("QQ-plot (¿puntos sobre la diagonal?)")
    sns.histplot(r, kde=True, ax=ax[1, 0]); ax[1, 0].set_title("Histograma de residuos")
    if "Fecha" in df: ax[1, 1].plot(df["Fecha"], r, lw=.6); ax[1, 1].axhline(0, c="r"); ax[1, 1].set_title("Residuos en el tiempo (¿ciclos?)")
    else: plot_acf(r, lags=40, ax=ax[1, 1])
    plt.tight_layout(); st.pyplot(f)
    f2, a2 = plt.subplots(figsize=(7, 2.6)); plot_acf(r, lags=40, ax=a2); a2.set_title("Autocorrelación de residuos"); st.pyplot(f2)
    sw = stats.shapiro(r).pvalue; bp = het_breuschpagan(r, X)[1]; dw = durbin_watson(r)
    tab = pd.DataFrame([
        ["Normalidad", "Shapiro-Wilk", f"p = {sw:.2e}", "✅ se cumple" if sw > .05 else "❌ no se cumple"],
        ["Varianza constante", "Breusch-Pagan", f"p = {bp:.2e}", "✅ se cumple" if bp > .05 else "❌ no se cumple"],
        ["Independencia", "Durbin-Watson", f"DW = {dw:.2f} (ideal ≈ 2)", "✅ se cumple" if 1.5 < dw < 2.5 else "❌ no se cumple"]],
        columns=["Supuesto", "Prueba", "Resultado", "Veredicto"])
    st.table(tab)
    simple("""
Con p-valor **menor a 0,05** se rechaza el supuesto. Con series diarias es muy común que falle la **independencia**:
si hoy hace calor, mañana probablemente también. Entonces el modelo **no es completamente válido**:
la recta sigue describiendo la tendencia, pero los tests y errores calculados con las fórmulas clásicas son menos confiables.
""")

# ---------------------------------------------------------------- 5
elif pagina.startswith("5"):
    st.title("5. Test de hipótesis sobre la pendiente")
    c1, c2 = st.columns(2)
    xn = c1.selectbox("X", num, index=idx("Temperatura_Abrigo_150cm", num))
    yopts = [c for c in num if c != xn]
    yn = c2.selectbox("Y", yopts, index=idx("Radiacion_Global", yopts))
    c3, c4 = st.columns(2)
    alpha = c3.select_slider("Nivel de significación α", [0.01, 0.05, 0.10], value=0.05)
    tipo = c4.radio("Tipo de test", ["Bilateral (β₁ ≠ 0)", "Cola derecha (β₁ > 0)", "Cola izquierda (β₁ < 0)"], horizontal=True)
    X = sm.add_constant(df[xn]); m = sm.OLS(df[yn], X).fit(); n = len(df); gl = n - 2
    b1 = m.params.iloc[1]; se = m.bse.iloc[1]; t = b1 / se
    if tipo.startswith("Bil"):
        crit = stats.t.ppf(1 - alpha / 2, gl); p = 2 * stats.t.sf(abs(t), gl); rech = abs(t) > crit; h1 = "β₁ ≠ 0"
    elif "derecha" in tipo:
        crit = stats.t.ppf(1 - alpha, gl); p = stats.t.sf(t, gl); rech = t > crit; h1 = "β₁ > 0"
    else:
        crit = stats.t.ppf(alpha, gl); p = stats.t.cdf(t, gl); rech = t < crit; h1 = "β₁ < 0"
    st.markdown(f"**H₀:** β₁ = 0 (no hay relación lineal)  ·  **H₁:** {h1}  ·  **α = {alpha}**")
    st.markdown("**Supuestos que usamos:** errores normales, independientes y con varianza constante (se verifican en el paso 4).")
    k = st.columns(4); k[0].metric("β̂₁", f"{b1:.4f}"); k[1].metric("Error estándar", f"{se:.4f}")
    k[2].metric("t observado", f"{t:.2f}"); k[3].metric("p-valor", f"{p:.2e}")
    xs = np.linspace(-max(6, abs(t) + 1), max(6, abs(t) + 1), 600); f, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(xs, stats.t.pdf(xs, gl), c="k")
    if tipo.startswith("Bil"):
        ax.fill_between(xs, stats.t.pdf(xs, gl), where=np.abs(xs) > crit, color="red", alpha=.35, label="zona de rechazo")
    elif "derecha" in tipo: ax.fill_between(xs, stats.t.pdf(xs, gl), where=xs > crit, color="red", alpha=.35, label="zona de rechazo")
    else: ax.fill_between(xs, stats.t.pdf(xs, gl), where=xs < crit, color="red", alpha=.35, label="zona de rechazo")
    ax.axvline(t, c="blue", ls="--", label=f"t observado = {t:.2f}"); ax.legend(); ax.set_title(f"Distribución t con {gl} grados de libertad bajo H₀"); st.pyplot(f)
    (st.error if rech else st.info)(f"**Decisión: {'SE RECHAZA H₀' if rech else 'NO se rechaza H₀'}** "
        f"(valor crítico = {crit:.3f}; p-valor {'<' if p < alpha else '≥'} α).")
    ic = m.conf_int(alpha=alpha).iloc[1]; st.write(f"**Intervalo de confianza {int((1-alpha)*100)} % para β₁:** [{ic[0]:.4f}, {ic[1]:.4f}]  (si no contiene el 0, es coherente con rechazar H₀ en el test bilateral)")
    simple(f"""
Suponemos primero que **no hay relación** (H₀: pendiente = 0) y medimos cuán *raro* sería obtener una pendiente como la nuestra **si eso fuera cierto**.
- El **estadístico t** mide cuántos errores estándar se aleja la pendiente de 0. Acá: {t:.2f}.
- Si cae en la **zona roja** (muy lejos de 0), es demasiado raro para ser casualidad → se rechaza H₀.
- El **p-valor** es la probabilidad de ver algo tan extremo si H₀ fuera cierta. Pequeño = evidencia contra H₀.

Probá cambiar **α** o el **tipo de test** y mirá cómo se mueve la zona roja.
""")
    st.subheader("¿Es válida la conclusión?")
    rob = sm.OLS(df[yn], X).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    st.write("Como los datos son diarios, los errores suelen estar correlacionados. Comparemos con un error estándar **robusto** (Newey-West):")
    st.table(pd.DataFrame({"Clásico": [f"{se:.4f}", f"{t:.2f}", f"{m.pvalues.iloc[1]:.2e}"],
                           "Robusto (HAC)": [f"{rob.bse.iloc[1]:.4f}", f"{rob.tvalues.iloc[1]:.2f}", f"{rob.pvalues.iloc[1]:.2e}"]},
                          index=["Error estándar", "t", "p-valor"]))
    simple("Si con el método robusto la conclusión **no cambia** (sigue siendo significativa), podemos confiar en que la relación existe, "
           "aunque con menos precisión de la que mostraba el test clásico.")

# ---------------------------------------------------------------- 6
else:
    st.title("6. Regresión lineal múltiple")
    yn = st.selectbox("Variable dependiente (Y)", num, index=idx("Radiacion_Global", num))
    cand = [c for c in num if c != yn]
    xs = st.multiselect("Variables explicativas", cand, default=cand)
    if not xs: st.warning("Elegí al menos una variable."); st.stop()
    X = sm.add_constant(df[xs]); m = sm.OLS(df[yn], X).fit()
    eq = " + ".join(f"{b:.3f}·{n}" for n, b in m.params.drop("const").items())
    st.success(f"**{yn}** = {m.params['const']:.2f} + {eq}")
    k = st.columns(4); k[0].metric("R²", f"{m.rsquared:.3f}"); k[1].metric("R² ajustado", f"{m.rsquared_adj:.3f}")
    k[2].metric("F global", f"{m.fvalue:.1f}"); k[3].metric("p (F)", f"{m.f_pvalue:.1e}")
    z = (df[xs + [yn]] - df[xs + [yn]].mean()) / df[xs + [yn]].std(); zb = sm.OLS(z[yn], z[xs]).fit().params
    tab = pd.DataFrame({"coeficiente": m.params, "error est.": m.bse, "t": m.tvalues, "p-valor": m.pvalues, "coef. estandarizado": zb})
    st.dataframe(tab.round(4))
    if len(xs) > 1:
        vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])], index=xs, name="VIF")
        st.write("**Multicolinealidad (VIF):** " + " · ".join(f"{k_}: {v_:.2f}" for k_, v_ in vif.items()) + "  (preocupa si supera ~5-10)")
    base = sm.OLS(df[yn], sm.add_constant(df[xs[0]])).fit()
    st.write(f"Comparación: con solo **{xs[0]}** el R² era {base.rsquared:.3f}; con todas las elegidas es **{m.rsquared:.3f}**.")
    f, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].scatter(m.fittedvalues, df[yn], alpha=.4, s=10); lim = [df[yn].min(), df[yn].max()]; ax[0].plot(lim, lim, "r"); ax[0].set(title="Observado vs. ajustado", xlabel="ajustado", ylabel="observado")
    ax[1].scatter(m.fittedvalues, m.resid, alpha=.4, s=10); ax[1].axhline(0, c="r"); ax[1].set_title("Residuos vs. ajustados")
    sm.qqplot(m.get_influence().resid_studentized_internal, line="45", ax=ax[2]); ax[2].set_title("QQ-plot")
    plt.tight_layout(); st.pyplot(f)
    st.write(f"Shapiro p = {stats.shapiro(m.resid).pvalue:.3f} · Breusch-Pagan p = {het_breuschpagan(m.resid, X)[1]:.2e} · Durbin-Watson = {durbin_watson(m.resid):.2f}")
    simple("""
Es lo mismo que la regresión simple, pero usando **varias pistas a la vez**. Cada coeficiente se lee así: *cuánto cambia Y al subir esa variable 1 unidad, manteniendo las demás constantes*.
- **R² ajustado:** como el R² siempre sube al agregar variables, el ajustado "castiga" las que no aportan.
- **Coef. estandarizado:** permite comparar cuál variable pesa más, aunque tengan escalas distintas (mayor valor absoluto = más influyente).
- **VIF:** detecta si dos variables explicativas dicen casi lo mismo entre sí.
- Fijate que un coeficiente puede **cambiar de signo** respecto de la correlación simple: pasa cuando una variable está relacionada con otras del modelo.

Probá **sacar y agregar variables** y mirá cómo cambian el R² y los coeficientes.
""")
