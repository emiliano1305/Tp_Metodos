import io, os
import numpy as np, pandas as pd, streamlit as st
import matplotlib.pyplot as plt, seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.tsaplots import plot_acf

st.set_page_config(page_title="Análisis estadístico", page_icon="📊", layout="wide")
sns.set_theme(style="whitegrid", font_scale=.85)

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(BASE, "01_Buenos_Aires_Castelar.csv")
LOGO_REPO = next((os.path.join(BASE, n) for n in ("logo.png", "logo.jpg", "logo.jpeg")
                  if os.path.exists(os.path.join(BASE, n))), None)

# ------------------------------------------------------------------ utilidades
@st.cache_data
def cargar(contenido: bytes):
    df = pd.read_csv(io.BytesIO(contenido), sep=None, engine="python")  # detecta ; o ,
    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    return df

def unir(items):
    items = list(items)
    if len(items) <= 1: return "".join(items)
    return ", ".join(items[:-1]) + " y " + items[-1]

def tarjeta(titulo, texto):
    with st.container(border=True):
        st.markdown(f"**{titulo}**")
        st.markdown(texto)

def fuerza(r):
    a = abs(r)
    return "fuerte" if a >= .7 else ("moderada" if a >= .4 else "débil")

def pruebas(m, X):
    r = m.resid
    sw = stats.shapiro(r).pvalue if len(r) >= 3 else np.nan
    return sw, het_breuschpagan(r, X)[1], durbin_watson(r)

def incumplidos(sw, bp, dw):
    out = []
    if sw <= .05: out.append("normalidad")
    if bp <= .05: out.append("homocedasticidad")
    if not (1.5 < dw < 2.5): out.append("independencia")
    return out

def idx(nombre, lista):
    return lista.index(nombre) if nombre in lista else 0

# ------------------------------------------------------------------ barra lateral
st.sidebar.title("📊 Análisis estadístico")
st.sidebar.caption("Métodos Estadísticos para Inteligencia Artificial")
sub = st.sidebar.file_uploader("Subir archivo para analizar (CSV)", type="csv")
nav = st.sidebar.container()
with st.sidebar.expander("Personalizar presentación"):
    logo_up = st.file_uploader("Logo (PNG o JPG)", type=["png", "jpg", "jpeg"])
LOGO = logo_up.getvalue() if logo_up else LOGO_REPO

def encabezado(titulo):
    c1, c2 = st.columns([5, 1])
    c1.title(titulo)
    if LOGO: c2.image(LOGO, width=120)

# ------------------------------------------------------------------ carga de datos
contenido = None
if sub is not None:
    contenido = sub.getvalue()
elif os.path.exists(DEFAULT_CSV):
    contenido = open(DEFAULT_CSV, "rb").read()

if contenido is None:
    encabezado("Análisis estadístico")
    st.info("Para comenzar, subir un archivo CSV desde el panel lateral (**Subir archivo para analizar**).")
    st.markdown("El análisis incluye estadística descriptiva, regresión lineal simple, análisis de residuos, "
                "test de hipótesis sobre la pendiente y regresión lineal múltiple. "
                "El archivo debe contener al menos dos columnas numéricas.")
    st.stop()

try:
    df = cargar(contenido)
except Exception:
    encabezado("Análisis estadístico")
    st.warning("El archivo no tiene un formato CSV interpretable. Se requiere una tabla con encabezados y columnas separadas por coma o punto y coma.")
    st.stop()

num = df.select_dtypes("number").columns.tolist()
if len(num) < 2:
    encabezado("Análisis estadístico")
    st.warning("El archivo debe contener al menos dos columnas numéricas para realizar el análisis.")
    st.stop()

has_fecha = "Fecha" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Fecha"]) and df["Fecha"].notna().any()
origen = sub.name if sub is not None else "Estación meteorológica Castelar (datos de ejemplo)"
nav.caption(f"Datos: {origen} · {len(df)} observaciones")
pagina = nav.radio("Secciones", [
    "1. Los datos", "2. Descripción numérica y gráfica", "3. Regresión simple",
    "4. Análisis de residuos", "5. Test de hipótesis de la pendiente", "6. Regresión múltiple"])

# ------------------------------------------------------------------ 1
if pagina.startswith("1"):
    encabezado("1. Los datos")
    faltantes = int(df.isna().sum().sum()); dup = int(df.duplicated().sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observaciones", len(df)); c2.metric("Variables", df.shape[1])
    c3.metric("Valores faltantes", faltantes); c4.metric("Filas duplicadas", dup)
    st.dataframe(df.head(15))
    st.subheader("Tipo de cada variable")
    st.dataframe(pd.DataFrame({"tipo": df.dtypes.astype(str), "faltantes": df.isna().sum(), "valores distintos": df.nunique()}))
    txt = (f"La base contiene **{len(df)} observaciones** y **{df.shape[1]} variables**, de las cuales **{len(num)} son cuantitativas** "
           f"({unir(num)}). ")
    if has_fecha:
        dif = df["Fecha"].diff().dropna()
        diario = (dif == pd.Timedelta(days=1)).all()
        txt += (f"Los datos abarcan desde el {df['Fecha'].min().date()} hasta el {df['Fecha'].max().date()}"
                + (", con una observación por día y sin interrupciones. " if diario else ". "))
    txt += ("No se registran valores faltantes ni filas duplicadas, por lo que no fue necesario imputar ni eliminar observaciones."
            if faltantes == 0 and dup == 0 else
            f"Se registran {faltantes} valores faltantes y {dup} filas duplicadas; en los análisis posteriores se excluyen las filas incompletas de las variables involucradas.")
    tarjeta("Interpretación", txt)

# ------------------------------------------------------------------ 2
elif pagina.startswith("2"):
    encabezado("2. Descripción numérica y gráfica")
    d = df[num]
    t = pd.DataFrame({"media": d.mean(), "mediana": d.median(), "mín": d.min(), "Q1": d.quantile(.25),
                      "Q3": d.quantile(.75), "máx": d.max(), "desvío": d.std(),
                      "CV %": d.std() / d.mean().abs() * 100, "asimetría": d.skew(), "curtosis": d.kurt()}).T
    st.subheader("Medidas de posición y dispersión")
    st.dataframe(t.round(3))
    cv = t.loc["CV %"].replace([np.inf, -np.inf], np.nan).dropna()
    txt = "Para cada variable se calcularon medidas de posición (media, mediana y cuartiles), de dispersión (desvío estándar y coeficiente de variación) y de forma (asimetría y curtosis). "
    if len(cv) >= 2:
        txt += (f"Según el coeficiente de variación, **{cv.idxmax()}** es la variable de mayor variabilidad relativa (CV = {cv.max():.1f} %) "
                f"y **{cv.idxmin()}** la más estable (CV = {cv.min():.1f} %).")
    tarjeta("Interpretación", txt)

    st.subheader("Distribución de una variable")
    v = st.selectbox("Variable a analizar", num, index=len(num) - 1)
    s = d[v].dropna()
    c1, c2 = st.columns([2, 1])
    f, ax = plt.subplots(figsize=(6, 3.2)); sns.histplot(s, kde=True, ax=ax, color="steelblue"); c1.pyplot(f)
    f, ax = plt.subplots(figsize=(3, 3.2)); sns.boxplot(y=s, ax=ax, color="lightsteelblue"); c2.pyplot(f)
    sk, ku = s.skew(), s.kurt()
    forma_s = "aproximadamente simétrica" if abs(sk) < .5 else ("asimétrica, con cola hacia valores altos" if sk > 0 else "asimétrica, con cola hacia valores bajos")
    forma_k = "achatada (curtosis marcadamente negativa)" if ku < -1 else ("con colas pesadas" if ku > 1 else "con apuntamiento similar al de una distribución normal")
    q1, q3 = s.quantile(.25), s.quantile(.75); iqr = q3 - q1
    n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
    p = stats.shapiro(s).pvalue
    norm = ("no se rechaza la hipótesis de normalidad" if p > .05 else "se rechaza la hipótesis de normalidad")
    tarjeta("Interpretación",
        f"**{v}:** media = {s.mean():.2f}, mediana = {s.median():.2f}, desvío estándar = {s.std():.2f}. "
        f"La distribución es {forma_s} y {forma_k}. "
        f"Se identificaron **{n_out}** valores atípicos según el criterio de Tukey (1,5·RIC). "
        f"En el test de Shapiro-Wilk (p = {p:.2e}) {norm}.")

    if has_fecha:
        st.subheader("Evolución en el tiempo")
        f, ax = plt.subplots(figsize=(11, 2.8)); ax.plot(df["Fecha"], df[v], lw=.8); ax.set_ylabel(v); st.pyplot(f)
        r1 = s.autocorr(1)
        dep = ("alta: las observaciones consecutivas son muy similares entre sí" if abs(r1) > .7
               else "moderada" if abs(r1) > .4 else "baja")
        tarjeta("Interpretación",
            f"La serie de **{v}** muestra su evolución día a día. La autocorrelación de orden 1 es {r1:.2f}, es decir, una dependencia temporal {dep}. "
            "Esta dependencia condiciona el supuesto de independencia de los errores en los modelos de regresión.")

    st.subheader("Correlaciones")
    f, ax = plt.subplots(figsize=(5.5, 4.2)); sns.heatmap(d.corr(), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax); st.pyplot(f)
    cm = d.corr()
    pares = [(a, b, cm.loc[a, b]) for i, a in enumerate(num) for b in num[i + 1:] if pd.notna(cm.loc[a, b])]
    if pares:
        a, b, r = max(pares, key=lambda z: abs(z[2]))
        sentido = "positiva (ambas variables tienden a aumentar juntas)" if r > 0 else "negativa (cuando una aumenta, la otra tiende a disminuir)"
        tarjeta("Interpretación",
            f"El coeficiente de correlación de Pearson mide la intensidad de la relación lineal entre dos variables (de −1 a +1). "
            f"La asociación más intensa se da entre **{a}** y **{b}** (r = {r:.2f}): una relación {fuerza(r)} y {sentido}.")

# ------------------------------------------------------------------ 3
elif pagina.startswith("3"):
    encabezado("3. Regresión lineal simple")
    c1, c2 = st.columns(2)
    xn = c1.selectbox("Variable explicativa (X)", num, index=idx("Temperatura_Abrigo_150cm", num))
    yopts = [c for c in num if c != xn]
    yn = c2.selectbox("Variable a explicar (Y)", yopts, index=idx("Radiacion_Global", yopts))
    dd = df[[xn, yn]].dropna()
    if len(dd) < 5: st.warning("No hay suficientes observaciones completas para ajustar el modelo."); st.stop()
    X = sm.add_constant(dd[xn]); y = dd[yn]; m = sm.OLS(y, X).fit()
    b0, b1 = m.params; r = np.corrcoef(dd[xn], y)[0, 1]
    st.success(f"**Recta ajustada:**  {yn} = {b0:.3f} + {b1:.3f} · {xn}")
    k1, k2, k3 = st.columns(3)
    k1.metric("Pendiente (β₁)", f"{b1:.3f}"); k2.metric("R²", f"{m.rsquared:.3f}"); k3.metric("Correlación r", f"{r:.3f}")
    f, ax = plt.subplots(figsize=(7, 4))
    sns.regplot(x=xn, y=yn, data=dd, scatter_kws=dict(alpha=.35, s=14), line_kws=dict(color="red"), ax=ax); st.pyplot(f)
    fuera = not (dd[xn].min() <= 0 <= dd[xn].max())
    inter = (f"El intercepto (β₀ = {b0:.3f}) corresponde a {xn} = 0, valor que queda fuera del rango observado, por lo que no tiene interpretación práctica."
             if fuera else f"El intercepto (β₀ = {b0:.3f}) es el valor esperado de {yn} cuando {xn} = 0.")
    tarjeta("Interpretación",
        f"Se ajustó por mínimos cuadrados un modelo lineal de **{yn}** en función de **{xn}** (n = {len(dd)}). "
        f"La pendiente estimada indica que, por cada unidad que aumenta {xn}, {yn} {'aumenta' if b1 > 0 else 'disminuye'} en promedio {abs(b1):.3f} unidades. {inter} "
        f"El coeficiente de determinación es R² = {m.rsquared:.3f}: el modelo explica el **{m.rsquared*100:.1f} %** de la variabilidad de {yn}, "
        f"mientras que el {(1-m.rsquared)*100:.1f} % restante se atribuye a otros factores no incluidos. "
        f"La correlación lineal es {fuerza(r)} (r = {r:.3f}).")

    SST = ((y - y.mean()) ** 2).sum(); SSE = (m.resid ** 2).sum()
    st.subheader("Coeficiente de determinación")
    st.latex(r"R^2=\frac{SSR}{SST}=1-\frac{SSE}{SST}")
    st.write(f"SST (variación total) = {SST:.1f} · SSR (variación explicada) = {SST-SSE:.1f} · SSE (variación no explicada) = {SSE:.1f}  →  R² = {(SST-SSE)/SST:.4f}")
    st.subheader("Estimación con el modelo ajustado")
    xv = st.slider(f"Valor de {xn}", float(dd[xn].min()), float(dd[xn].max()), float(dd[xn].mean()))
    st.write(f"Para {xn} = {xv:.2f}, el modelo estima {yn} = {b0:.3f} + {b1:.3f} × {xv:.2f} = **{b0 + b1*xv:.2f}**")
    st.subheader("Tabla de coeficientes")
    st.dataframe(m.summary2().tables[1].round(4))

# ------------------------------------------------------------------ 4
elif pagina.startswith("4"):
    encabezado("4. Análisis de residuos")
    c1, c2 = st.columns(2)
    xn = c1.selectbox("Variable explicativa (X)", num, index=idx("Temperatura_Abrigo_150cm", num))
    yopts = [c for c in num if c != xn]
    yn = c2.selectbox("Variable a explicar (Y)", yopts, index=idx("Radiacion_Global", yopts))
    dd = df[[xn, yn]].dropna()
    if len(dd) < 5: st.warning("No hay suficientes observaciones completas para ajustar el modelo."); st.stop()
    X = sm.add_constant(dd[xn]); m = sm.OLS(dd[yn], X).fit(); r = m.resid; fit = m.fittedvalues
    tarjeta("Qué se evalúa",
        "El residuo es la diferencia entre el valor observado y el valor predicho por la recta. Si el modelo es adecuado, los residuos se comportan como ruido aleatorio. Se analizaron:\n\n"
        "- **Residuos vs. ajustados:** permite detectar curvatura (falta de linealidad) o forma de embudo (varianza no constante).\n"
        "- **QQ-plot e histograma:** comparan la distribución de los errores con la normal.\n"
        "- **Residuos en el tiempo y autocorrelación:** permiten detectar dependencia entre observaciones consecutivas.")
    s = m.get_influence().resid_studentized_internal
    f, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].scatter(fit, r, alpha=.4, s=12); ax[0, 0].axhline(0, c="r"); ax[0, 0].set(title="Residuos vs. ajustados", xlabel="ajustado")
    sm.qqplot(s, line="45", ax=ax[0, 1]); ax[0, 1].set_title("QQ-plot (residuos estudentizados)")
    sns.histplot(r, kde=True, ax=ax[1, 0]); ax[1, 0].set_title("Histograma de residuos")
    if has_fecha:
        ax[1, 1].plot(df.loc[dd.index, "Fecha"], r, lw=.6); ax[1, 1].axhline(0, c="r"); ax[1, 1].set_title("Residuos en el tiempo")
    else:
        ax[1, 1].plot(r.values, lw=.6); ax[1, 1].axhline(0, c="r"); ax[1, 1].set_title("Residuos por observación")
    plt.tight_layout(); st.pyplot(f)
    f2, a2 = plt.subplots(figsize=(7, 2.6)); plot_acf(r, lags=min(40, len(r) // 2 - 1), ax=a2); a2.set_title("Autocorrelación de residuos"); st.pyplot(f2)
    sw, bp, dw = pruebas(m, X); fallan = incumplidos(sw, bp, dw)
    st.table(pd.DataFrame([
        ["Normalidad", "Shapiro-Wilk", f"p = {sw:.2e}", "✅ se cumple" if sw > .05 else "❌ no se cumple"],
        ["Varianza constante", "Breusch-Pagan", f"p = {bp:.2e}", "✅ se cumple" if bp > .05 else "❌ no se cumple"],
        ["Independencia", "Durbin-Watson", f"DW = {dw:.2f} (ideal ≈ 2)", "✅ se cumple" if 1.5 < dw < 2.5 else "❌ no se cumple"]],
        columns=["Supuesto", "Prueba", "Resultado", "Veredicto"]))
    r1 = pd.Series(r.values).autocorr(1)
    if not fallan:
        concl = "Los tres supuestos se cumplen, por lo que el modelo es válido y las inferencias (tests e intervalos de confianza) son confiables."
    else:
        concl = (f"No {'se cumple' if len(fallan) == 1 else 'se cumplen'} **{unir(fallan)}**. La autocorrelación de orden 1 de los residuos es {r1:.2f}. "
                 "En consecuencia, el modelo describe correctamente la tendencia, pero no es plenamente válido en sentido inferencial: "
                 "los errores estándar, los tests t y los intervalos calculados con las fórmulas clásicas son solo aproximados. "
                 "Para contrastar la conclusión se recalcularon con errores estándar robustos (sección 5).")
    tarjeta("Conclusión sobre la validez del modelo", concl)

# ------------------------------------------------------------------ 5
elif pagina.startswith("5"):
    encabezado("5. Test de hipótesis sobre la pendiente")
    c1, c2 = st.columns(2)
    xn = c1.selectbox("Variable explicativa (X)", num, index=idx("Temperatura_Abrigo_150cm", num))
    yopts = [c for c in num if c != xn]
    yn = c2.selectbox("Variable a explicar (Y)", yopts, index=idx("Radiacion_Global", yopts))
    c3, c4 = st.columns(2)
    alpha = c3.select_slider("Nivel de significación α", [0.01, 0.05, 0.10], value=0.05)
    tipo = c4.radio("Tipo de test", ["Bilateral (β₁ ≠ 0)", "Cola derecha (β₁ > 0)", "Cola izquierda (β₁ < 0)"], horizontal=True)
    dd = df[[xn, yn]].dropna()
    if len(dd) < 5: st.warning("No hay suficientes observaciones completas para ajustar el modelo."); st.stop()
    X = sm.add_constant(dd[xn]); m = sm.OLS(dd[yn], X).fit(); n = len(dd); gl = n - 2
    b1 = m.params.iloc[1]; se = m.bse.iloc[1]; t = b1 / se
    if tipo.startswith("Bil"):
        crit = stats.t.ppf(1 - alpha / 2, gl); p = 2 * stats.t.sf(abs(t), gl); rech = abs(t) > crit; h1 = "β₁ ≠ 0"
        concl_h1 = f"existe relación lineal entre {xn} y {yn}"; regla = f"|t| > {crit:.3f}"
    elif "derecha" in tipo:
        crit = stats.t.ppf(1 - alpha, gl); p = stats.t.sf(t, gl); rech = t > crit; h1 = "β₁ > 0"
        concl_h1 = f"la relación lineal entre {xn} y {yn} es positiva"; regla = f"t > {crit:.3f}"
    else:
        crit = stats.t.ppf(alpha, gl); p = stats.t.cdf(t, gl); rech = t < crit; h1 = "β₁ < 0"
        concl_h1 = f"la relación lineal entre {xn} y {yn} es negativa"; regla = f"t < {crit:.3f}"

    st.markdown("#### Planteo del test")
    st.markdown(f"- **Hipótesis nula:** H₀: β₁ = 0 (no hay relación lineal entre {xn} y {yn}).\n"
                f"- **Hipótesis alternativa:** H₁: {h1}.\n"
                f"- **Nivel de significación:** α = {alpha}.\n"
                f"- **Supuestos:** errores normales, independientes y de varianza constante (verificados en la sección 4).\n"
                f"- **Estadístico:** t = β̂₁ / EE(β̂₁), que bajo H₀ sigue una distribución t con {gl} grados de libertad.\n"
                f"- **Regla de decisión:** se rechaza H₀ si {regla}.")
    st.markdown("#### Resultado")
    k = st.columns(4); k[0].metric("β̂₁", f"{b1:.4f}"); k[1].metric("Error estándar", f"{se:.4f}")
    k[2].metric("t observado", f"{t:.2f}"); k[3].metric("p-valor", f"{p:.2e}")
    xs = np.linspace(-max(6, abs(t) + 1), max(6, abs(t) + 1), 600); f, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(xs, stats.t.pdf(xs, gl), c="k")
    zona = np.abs(xs) > crit if tipo.startswith("Bil") else (xs > crit if "derecha" in tipo else xs < crit)
    ax.fill_between(xs, stats.t.pdf(xs, gl), where=zona, color="red", alpha=.35, label="zona de rechazo")
    ax.axvline(t, c="blue", ls="--", label=f"t observado = {t:.2f}"); ax.legend(); ax.set_title(f"Distribución t con {gl} grados de libertad bajo H₀"); st.pyplot(f)
    ic = m.conf_int(alpha=alpha).iloc[1]
    if rech:
        st.error(f"**Decisión: se rechaza H₀** (p-valor {p:.2e} < α = {alpha}). Con un nivel de significación de {alpha}, hay evidencia estadística de que {concl_h1}.")
    else:
        st.info(f"**Decisión: no se rechaza H₀** (p-valor {p:.2e} ≥ α = {alpha}). Los datos no aportan evidencia suficiente para afirmar que {concl_h1}.")
    tarjeta("Interpretación",
        f"El estadístico t = {t:.2f} indica que la pendiente estimada ({b1:.4f}) se ubica a {abs(t):.1f} errores estándar de cero. "
        f"El p-valor es la probabilidad de obtener un resultado al menos tan extremo si H₀ fuera cierta. "
        f"El intervalo de confianza del {int((1-alpha)*100)} % para β₁ es [{ic[0]:.4f}, {ic[1]:.4f}]"
        + (", que no contiene el cero, en coincidencia con el rechazo de H₀." if (ic[0] > 0 or ic[1] < 0) else ", que contiene el cero."))

    st.markdown("#### Validez de la conclusión")
    sw, bp, dw = pruebas(m, X); fallan = incumplidos(sw, bp, dw)
    rob = sm.OLS(dd[yn], X).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    st.table(pd.DataFrame({"Clásico": [f"{se:.4f}", f"{t:.2f}", f"{m.pvalues.iloc[1]:.2e}"],
                           "Robusto (Newey-West)": [f"{rob.bse.iloc[1]:.4f}", f"{rob.tvalues.iloc[1]:.2f}", f"{rob.pvalues.iloc[1]:.2e}"]},
                          index=["Error estándar", "t", "p-valor (bilateral)"]))
    sig_c = m.pvalues.iloc[1] < alpha; sig_r = rob.pvalues.iloc[1] < alpha
    if fallan:
        base = (f"El test clásico exige normalidad, independencia y varianza constante de los errores. En el análisis de residuos no {'se cumple' if len(fallan) == 1 else 'se cumplen'} **{unir(fallan)}**, "
                f"por lo que el p-valor clásico es aproximado. Como control, se recalculó el error estándar con el estimador robusto de Newey-West (de {se:.4f} a {rob.bse.iloc[1]:.4f}). ")
    else:
        base = "Los supuestos del test se cumplen, por lo que el test clásico es válido. Como control adicional, se recalculó con el estimador robusto de Newey-West. "
    base += ("Con el método robusto la pendiente continúa siendo significativa: **la conclusión se mantiene**, aunque con menor precisión que la sugerida por el test clásico."
             if (sig_c and sig_r) else
             "Con el método robusto la pendiente no resulta significativa: **la conclusión del test clásico no se sostiene**." if (sig_c and not sig_r) else
             "Ambos métodos coinciden en que la pendiente no es significativa." if (not sig_c and not sig_r) else
             "Con el método robusto la pendiente resulta significativa, a diferencia del test clásico.")
    tarjeta("Interpretación", base + " La asociación estadística no implica causalidad.")

# ------------------------------------------------------------------ 6
else:
    encabezado("6. Regresión lineal múltiple")
    yn = st.selectbox("Variable dependiente (Y)", num, index=idx("Radiacion_Global", num))
    cand = [c for c in num if c != yn]
    xs = st.multiselect("Variables explicativas", cand, default=cand)
    if not xs: st.warning("Seleccionar al menos una variable explicativa."); st.stop()
    dd = df[xs + [yn]].dropna()
    if len(dd) < len(xs) + 5: st.warning("No hay suficientes observaciones completas para ajustar el modelo."); st.stop()
    X = sm.add_constant(dd[xs]); m = sm.OLS(dd[yn], X).fit()
    eq = " + ".join(f"{b:.3f}·{nm}" for nm, b in m.params.drop("const").items())
    st.success(f"**{yn}** = {m.params['const']:.2f} + {eq}")
    k = st.columns(4); k[0].metric("R²", f"{m.rsquared:.3f}"); k[1].metric("R² ajustado", f"{m.rsquared_adj:.3f}")
    k[2].metric("F global", f"{m.fvalue:.1f}"); k[3].metric("p (F)", f"{m.f_pvalue:.1e}")
    z = (dd - dd.mean()) / dd.std(); zb = sm.OLS(z[yn], z[xs]).fit().params
    st.dataframe(pd.DataFrame({"coeficiente": m.params, "error est.": m.bse, "t": m.tvalues, "p-valor": m.pvalues, "coef. estandarizado": zb}).round(4))
    base = sm.OLS(dd[yn], sm.add_constant(dd[xs[0]])).fit()
    f, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].scatter(m.fittedvalues, dd[yn], alpha=.4, s=10); lim = [dd[yn].min(), dd[yn].max()]; ax[0].plot(lim, lim, "r"); ax[0].set(title="Observado vs. ajustado", xlabel="ajustado", ylabel="observado")
    ax[1].scatter(m.fittedvalues, m.resid, alpha=.4, s=10); ax[1].axhline(0, c="r"); ax[1].set_title("Residuos vs. ajustados")
    sm.qqplot(m.get_influence().resid_studentized_internal, line="45", ax=ax[2]); ax[2].set_title("QQ-plot")
    plt.tight_layout(); st.pyplot(f)
    sw, bp, dw = pruebas(m, X); fallan = incumplidos(sw, bp, dw)
    sig = [v for v in xs if m.pvalues[v] < .05]
    mejor = zb.abs().idxmax()
    txt = (f"Se ajustó un modelo lineal múltiple de **{yn}** con {len(xs)} variable(s) explicativa(s) (n = {len(dd)}). "
           f"El modelo explica el **{m.rsquared*100:.1f} %** de la variabilidad (R² ajustado = {m.rsquared_adj:.3f}) y es globalmente significativo según el test F (p = {m.f_pvalue:.1e}). "
           f"Resultan individualmente significativas (p < 0,05): {unir(sig) if sig else 'ninguna'}. "
           f"Cada coeficiente expresa el cambio esperado en {yn} ante un aumento de una unidad de la variable, manteniendo constantes las demás. "
           f"Según los coeficientes estandarizados, la variable de mayor influencia es **{mejor}** (β* = {zb[mejor]:.3f}). ")
    txt += (f"Respecto del modelo simple con {xs[0]} (R² = {base.rsquared:.3f}), el R² {'aumenta' if m.rsquared >= base.rsquared else 'disminuye'} "
            f"en {abs(m.rsquared - base.rsquared)*100:.1f} puntos porcentuales. ")
    cambios = [v for v in xs if np.sign(np.corrcoef(dd[v], dd[yn])[0, 1]) != np.sign(zb[v])]
    if cambios:
        txt += (f"En {unir(cambios)}, el signo del coeficiente difiere del de su correlación simple con {yn}; "
                "esto refleja la relación de esa variable con las restantes variables explicativas del modelo. ")
    tarjeta("Interpretación del modelo", txt)
    if len(xs) > 1:
        vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])], index=xs)
        vtxt = " · ".join(f"{a}: {b:.2f}" for a, b in vif.items())
        conc = ("Todos los VIF son menores a 5: no se observan problemas de multicolinealidad." if vif.max() < 5
                else f"El VIF máximo es {vif.max():.2f}: existe multicolinealidad apreciable entre variables explicativas.")
        tarjeta("Multicolinealidad (VIF)", f"{vtxt}\n\n{conc}")
    if not fallan:
        rtxt = "Los residuos cumplen normalidad, varianza constante e independencia."
    else:
        rtxt = f"En los residuos no {'se cumple' if len(fallan) == 1 else 'se cumplen'} **{unir(fallan)}**; las inferencias del modelo deben considerarse aproximadas."
    tarjeta("Análisis de residuos",
            f"Shapiro-Wilk p = {sw:.3f} · Breusch-Pagan p = {bp:.2e} · Durbin-Watson = {dw:.2f}\n\n{rtxt}")
