# Cuando el modelo más complejo no es el que detecta mejor

*Comparativa entre LSTM, Transformer y VAE-LSTM sobre 19.000 vuelos reales de Barajas. Y por qué la pérdida de validación no es buena consejera.*

![Torre de control con tráfico aéreo sobre la pista al atardecer.](../docs/assets/tower-day.png)

> **Código:** [github.com/iRuperth/SADAR](https://github.com/iRuperth/SADAR)
> **Demo en vivo:** [huggingface.co/spaces/devrup404/sadar](https://huggingface.co/spaces/devrup404/sadar)
>
> El repositorio es público y está abierto a colaboraciones, sugerencias y forks. Cualquier discusión técnica, *issue* o *pull request* es bienvenida.

---

El dato más interesante del dataset no es un secuestro. Es un avión que fingió serlo durante treinta segundos por error.

El 31 de julio de 2017, un Airbus de Iberia voló sobre Madrid con el código de secuestro activado en el transpondedor durante tres puntos de su trayectoria. Después volvió a la normalidad. El vuelo era el IBE2845, el código transmitido era el 7500, y en el sistema secundario de vigilancia ese código significa interferencia ilícita.

No era un secuestro. Era casi con total seguridad un fallo del transpondedor: tres puntos sobre los 171 que componen el registro completo del vuelo, a 800 metros de la pista, sin más anomalía operativa. Aquel vuelo, anotado en un fichero parquet de 3,4 millones de filas, marca el inicio de este artículo. Y marca también la pregunta técnica que me llevé al proyecto: ¿puede un modelo aprender qué aspecto tiene un vuelo normal y avisar cuando uno deja de parecerlo?

Este artículo presenta SADAR, un sistema de detección de anomalías en trayectorias aéreas entrenado sobre dieciocho días de tráfico real procedente del aeropuerto Adolfo Suárez Madrid-Barajas. Compara tres autoencoders profundos contra un baseline clásico, evalúa el sistema sobre un banco de anomalías sintéticas y discute con franqueza dónde funciona y dónde no.

Empecemos por el problema.

## Planteamiento

La detección de incidentes en aviación civil rara vez puede formularse como un problema supervisado: los datos están dominados por operación normal y los pocos eventos etiquetados son ambiguos. La alternativa estándar es la **monitorización de conformidad**: aprender la distribución del comportamiento normal y señalar las trayectorias que se apartan de ella. Esta es la formulación que adopta SADAR.

La distinción importa. Decir "este vuelo se aparta del patrón aprendido" es defendible con datos. Decir "este vuelo es un incidente" requiere contexto operacional que el ADS-B no proporciona. SADAR opera estrictamente en el primer enunciado.

## Datos

Los datos provienen de OpenSky Network, una red colaborativa de receptores ADS-B. El conjunto cubre 18 días no consecutivos entre junio de 2017 y marzo de 2020, centrados en el espacio aéreo de Barajas. Cada aeronave reporta una posición cada 10 segundos, con 21 columnas por punto: posición y dinámica (latitud, longitud, altitud barométrica y geométrica, velocidad, rumbo, régimen vertical), identidad (callsign, icao24, flight_id) y estado del transpondedor (squawk, alert, onground).

Las cifras crudas, tras la unión y deduplicación:

- 3.429.638 filas.
- 19.057 vuelos distintos.
- Mediana de 175 puntos por vuelo, máximo 1.606.
- Mediana de duración de 30,2 minutos.

El primer hallazgo del análisis exploratorio fueron los huecos. La altitud GPS (`geoaltitude`) presenta un 24,79 % de valores ausentes. La velocidad y el rumbo, alrededor del 13 %. Estos huecos no son errores: corresponden a zonas de menor cobertura del receptor terrestre. Descarté la altitud GPS y me quedé con la barométrica, más completa, tratando los huecos como una propiedad del problema y no como un fallo a corregir.

![Trayectorias muestreadas sobre el espacio aéreo de LEMD. Los corredores de aproximación y salida se identifican visualmente antes de cualquier proyección. En rojo, los puntos de los cuatro vuelos con squawk de emergencia.](../docs/assets/eda/trajectories_map.png)

El segundo hallazgo provino del histograma del rumbo. Representado en un gráfico polar, las dos pistas paralelas de Barajas aparecen como dos lóbulos opuestos.

![Histograma polar del rumbo sobre el dataset completo. Los dos lóbulos opuestos corresponden a las pistas paralelas de Barajas. Esta propiedad circular motivó la codificación del rumbo como par (sin, cos).](../docs/assets/eda/heading_polar.png)

El aeropuerto es identificable en los datos antes de proyectarlos sobre un mapa.

Esa propiedad circular del rumbo motivó su codificación como par seno y coseno: para un autoencoder, 359° y 1° deben quedar adyacentes, y una codificación lineal lo impide.

![Distribución de filas por fase de vuelo en el dataset. La fase de descenso domina, seguida de ascenso y operaciones en superficie, lo que condiciona qué patrones aprende mejor el modelo.](../docs/assets/eda/flight_phases.png)

## Eventos reales en el conjunto

El dataset contiene únicamente cuatro vuelos con código de emergencia en el transpondedor. Conviene describirlos con detalle, porque su escasez determina el planteamiento del problema:

- **IBE2845** (Iberia, 31 de julio de 2017). Squawk 7500 durante tres puntos sobre 171, transmitido a 800 metros de pista. Compatible con un fallo del transpondedor.
- **ELY395** (El Al, 2 de octubre de 2017). Squawk 7700 durante cinco puntos, declarado a 800 metros de pista. El avión aterrizó. Es el caso más plausible de emergencia real.
- **RYR61AD** (Ryanair, 28 de enero de 2019). Squawk 7600 (fallo de comunicaciones). Un punto. Aterrizó.
- **SWR202Y** (Swiss, 3 de febrero de 2020). Squawk 7700. Un punto. Aterrizó.

Cuatro etiquetas en un universo de no etiquetas. Esta proporción descarta el aprendizaje supervisado: ningún clasificador se entrena con cuatro positivos. Los cuatro vuelos se reservaron como conjunto de validación cualitativa, agrupados en 1.076 ventanas etiquetadas en `data/processed/anomalies.npy`, y no aparecen en ningún momento del entrenamiento.

## Metodología

### Aprendizaje de una sola clase

Sin etiquetas suficientes, el planteamiento es **aprendizaje de una sola clase** mediante autoencoders. La red aprende a reconstruir trayectorias normales. El error de reconstrucción opera como score de anomalía: una entrada parecida a las del entrenamiento produce un error bajo; una entrada inusual lo eleva.

Entrené cuatro detectores bajo idénticas condiciones (mismo preprocesado, mismo split, mismo umbral) para permitir una comparación rigurosa:

| # | Modelo | Idea | Pesos |
|---|---|---|---|
| 0 | Isolation Forest | Estadísticas resumen por ventana, no secuencial | baseline |
| 1 | LSTM autoencoder | Reconstrucción secuencial | 224 KB |
| 2 | Transformer autoencoder | Atención sobre la ventana completa | 635 KB |
| 3 | VAE-LSTM | Codificación probabilística del espacio latente | 222 KB |

### Preprocesado

Cuatro decisiones de preprocesado merecen mención específica porque determinan la calidad final de un detector de trayectorias.

**Coordenadas métricas relativas a la pista.** Latitud y longitud en grados resultan inadecuadas para modelar movimiento sobre un aeropuerto: las distancias dependen de la latitud, y el modelo no incorpora geodesia. El pipeline proyecta las coordenadas a UTM zona 30 (EPSG:32630), centradas en la pista de Barajas, y opera en metros.

**Rumbo como par (sin, cos).** Variable circular que la codificación lineal no representa con fidelidad. La codificación trigonométrica asegura que valores próximos en el círculo lo sean también en el espacio de features.

**Split por fecha y por flight_id.** Entrenamiento sobre 2017-2019, validación y test sobre 2020. Cada vuelo aparece en un único split. Esto bloquea simultáneamente la fuga temporal y la de instancia.

**Limpieza del conjunto de entrenamiento.** Excluí del entrenamiento los 100 vuelos con patrón de go-around y los cuatro con squawk de emergencia. El conjunto de entrenamiento debe representar la versión más conservadora de "vuelo normal". Mantener go-arounds dentro hace que el modelo aprenda a reconstruirlos sin dificultad, anulando su valor como anomalías.

Las ventanas finales tienen 60 pasos de 10 segundos (10 minutos por ventana), con solapamiento del 50 %. Siete features por paso:

```
[x_rel, y_rel, baroaltitude, velocity, sin_hdg, cos_hdg, vertrate]
```

El conjunto resultante: 61.008 ventanas de entrenamiento, 7.679 de validación, 7.788 de test y 1.076 etiquetadas como anómalas. El escalador estandariza las features ajustándose únicamente sobre el conjunto de entrenamiento.

### Arquitecturas

**Isolation Forest** sobre estadísticas resumen (media, desviación, mínimo y máximo por feature). Baseline no secuencial cuya función es establecer la línea de referencia que el deep learning debe superar.

**LSTM autoencoder**: encoder con una capa LSTM (hidden 64), cuello de botella lineal a un latente de dimensión 16, decoder simétrico que desenrolla desde el latente repetido.

**Transformer autoencoder**: dos capas de encoder con cuatro cabezas de atención (d_model 64), codificación posicional sinusoidal, mismo cuello de botella a latente 16.

**VAE-LSTM**: encoder LSTM que produce la media y el logaritmo de la varianza, reparametrización para que los gradientes fluyan a través del muestreo, decoder LSTM desde z. La pérdida combina el error de reconstrucción (MSE) con la divergencia de Kullback-Leibler ponderada por un coeficiente beta de 0,0001193, seleccionado por Optuna.

```python
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + std * eps
```

Todos los modelos comparten el mismo loop de entrenamiento: optimizador Adam con weight decay, scheduler ReduceLROnPlateau, early stopping con paciencia 7, semilla fija a 42, tracking completo con MLflow.

### Banco de anomalías sintéticas

Cuatro eventos reales no permiten una evaluación cuantitativa estable. Para suplir esta limitación, se inyectan anomalías controladas sobre vuelos normales del conjunto de test. Cinco familias, todas con rampa lineal de inicio para permitir la medición de la **latencia de detección** (segundos desde el inicio de la perturbación hasta el primer cruce del umbral):

- **Desviación de ruta**: deformación lateral, magnitudes de 20, 40 y 80 km.
- **Anomalía de altitud**: offset de 300, 800 o 1.500 m.
- **Anomalía de velocidad**: factores multiplicativos de 0,4x, 1,6x y 2,2x.
- **Holding**: giros continuos con período de 120 o 240 segundos.
- **Congelación del transpondedor**: el ADS-B retiene el último valor durante el resto de la ventana.

La validación contra un simulador propio podría parecer circular. Es una práctica estándar en *safety engineering*: el simulador se diseñó antes que los modelos y describe modos de fallo bien definidos, no patrones aprendidos del modelo. El conjunto produce 12 variantes (combinaciones de tipo e intensidad) y 24.000 ventanas sintéticas por modelo evaluado.

Hasta aquí, el planteamiento. Lo siguiente es ver si funciona.

## Resultados

Resultados sobre el conjunto de test de 2020:

| Modelo | ROC-AUC real | PR-AUC real | ROC sintético | Latencia mediana |
|---|---|---|---|---|
| Isolation Forest | 0,515 | 0,133 | 0,593 | n/d |
| LSTM-AE | 0,648 | 0,260 | 0,779 | 115 s |
| Transformer-AE | 0,614 | 0,227 | 0,743 | 115 s |
| **VAE-LSTM** | **0,659** | **0,299** | **0,792** | 120 s |

Los modelos profundos prácticamente duplican la PR-AUC del baseline. Esta diferencia justifica la inversión en deep learning: si el LSTM no hubiera batido al Isolation Forest, el resto del trabajo no habría tenido sentido.

![Curva precision-recall del VAE-LSTM sobre el conjunto de test real. La PR-AUC de 0,299 duplica al baseline en condiciones de fuerte desbalance.](../reports/figures/pr_vae-lstm.png)

![Curva ROC del VAE-LSTM sobre el mismo conjunto. ROC-AUC de 0,659.](../reports/figures/roc_vae-lstm.png)

Una observación merece atención.

El Transformer ganó la pérdida de validación (0,038 frente a 0,053 del VAE y 0,072 del LSTM). Y perdió la PR-AUC. Algo no cuadraba.

La explicación es directa: una arquitectura con suficiente capacidad reconstruye también con cierta fidelidad las anomalías, lo que reduce la separabilidad entre normal y anómalo medida por el error de reconstrucción. En detección por reconstrucción, la pérdida de validación es un mal proxy del rendimiento. Lo sabemos y aun así la seguimos mirando.

Evalué además dos ensembles, por promedio y por máximo de los scores z-normalizados. Ninguno superó al VAE individual (PR-AUC 0,273 y 0,278 frente a 0,299), y los descarté. El ensemble fue un intento de cerrar con un empujón. No lo dio. Es honesto reportarlo.

Los números cuentan una parte. La otra está en lo que no sale en la tabla.

## Discusión

Los resultados muestran un perfil de detección **heterogéneo por tipo de anomalía**. El modelo final detecta con facilidad las perturbaciones ruidosas: ROC-AUC 0,984 en holding con periodo de 120 s, 0,989 en aceleración con factor 2,2, 0,899 en desviación de ruta de 80 km. Las perturbaciones sutiles (300 m de altitud, derivas pequeñas) requieren tiempos cercanos a los dos minutos y bajan al rango 0,51-0,55.

El caso más relevante de la discusión es la **congelación del transpondedor**.

El VAE lo detecta en el 1,2 % de los casos. Uno de cada ochenta intentos.

La explicación es estructural: un avión en crucero estable cuyos valores se congelan produce un patrón muy próximo al de un avión en crucero estable que continúa transmitiendo. El error de reconstrucción permanece bajo y el score no llega a cruzar el umbral. Es una limitación del enfoque de reconstrucción, no del entrenamiento.

![Matriz de confusión del VAE-LSTM con umbral en el percentil 99 de la validación. El balance favorece la precisión sobre la sensibilidad, consistente con una tasa de falsa alarma objetivo del 1 %.](../reports/figures/cm_vae-lstm.png)

Esta heterogeneidad delimita el alcance del sistema. SADAR es un detector de **maniobras anómalas**, no un detector universal de incidentes. Su valor operativo está en aquellas desviaciones que la red de reglas tradicional (sistemas tipo STCA o MSAW) no cubre, no en sustituir esos sistemas.

Antes de cerrar, conviene ser explícito en lo que el sistema no hace.

## Limitaciones

El sistema asume ciertas limitaciones documentadas explícitamente en el informe técnico.

- **Ausencia de planes de vuelo**. SADAR no dispone de la ruta prevista de cada vuelo. La "desviación respecto a la ruta esperada" se aproxima por "desviación respecto al patrón medio". Un vuelo que se aparte de su plan asignado pero permanezca dentro del patrón medio no será marcado.
- **Cobertura temporal**. El conjunto consta de 18 días no consecutivos entre 2017 y 2020. Modelar demanda, estacionalidad o predicción temporal continua queda fuera del alcance.
- **Congelación del transpondedor**. Tasa de detección dentro de ventana del 1,2 %. Cualquier extensión seria del sistema debe atacar este caso de forma específica (señales derivadas, detección de cambio de régimen, modelo dedicado).
- **Huecos de cobertura**. El 24,2 % de los vuelos en aire presentan gaps superiores a 120 segundos en la recepción. Estos huecos son artefactos del receptor, no incidentes operativos. Tratarlos como etiquetas positivas generaría ruido sistemático.

## Sistema desplegado

El modelo final se sirve a través de una API construida sobre FastAPI que monta un frontend React + Vite + TypeScript en el mismo puerto, lo que permite un despliegue *single-image* en Hugging Face Spaces. El dashboard consta de tres pantallas:

1. **Tower console**: radar en vivo sobre Barajas con los vuelos del día y sus scores, alerta visual al cruzar el umbral.
2. **Simulator**: inyección controlada de anomalías sobre un vuelo de referencia, con visualización de la trayectoria deformada, el score y la latencia.
3. **Metrics**: tabla comparativa de modelos, curvas precision-recall y ROC, matrices de confusión, desglose por tipo de anomalía.

## Stack y reproducibilidad

Python 3.11, PyTorch, FastAPI, MLflow y Optuna en el backend. Vite, React 18 y TypeScript en el frontend. Empaquetado en Docker *single-image*. Reproducibilidad garantizada por semillas fijas, configuraciones YAML que resuelven variables de entorno, y los checkpoints almacenados también como artifacts de MLflow.

## Conclusiones y trabajo futuro

SADAR demuestra que la monitorización de conformidad de trayectorias aéreas es abordable mediante autoencoders profundos entrenados exclusivamente sobre operación normal, con métricas defendibles y un protocolo de evaluación reproducible. La comparación entre tres familias de modelos bajo idénticas condiciones permite justificar la elección del VAE-LSTM no solo por su PR-AUC final, sino por su perfil global de detección y latencia.

El Transformer no fue la elección correcta para este problema. Lo dejo documentado porque los descartes también cuentan.

El trabajo futuro inmediato apunta a tres líneas. Primero, sustituir el Transformer por un TCN o un autoencoder convolucional para acotar la capacidad sin perder modelado temporal. Segundo, incorporar derivadas y detección de cambio de régimen para atacar la congelación del transpondedor. Tercero, evaluar la incorporación de planes de vuelo cuando estén disponibles, lo que replantearía el problema como predicción condicionada por el plan en lugar de aproximación al patrón medio.

El IBE2845 aterrizó sin incidentes el 31 de julio de 2017 y sus tres puntos en rojo no recibieron mayor atención. Glitch, anomalía o incidente, la realidad rara vez llega etiquetada, y construir sistemas que operen en ese terreno empieza por aceptarlo.
