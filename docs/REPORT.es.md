# SADAR

**Smart Anomaly Detection for Aviation Routes**

Memoria técnica

---

**Proyecto:** SADAR · Flight Conformance Monitor
**Dominio:** Vigilancia de tráfico aéreo · datos ADS-B
**Aeropuerto de estudio:** Madrid-Barajas (LEMD)
**Enfoque:** Aprendizaje profundo de una sola clase para detección de anomalías en trayectorias
**Fecha:** 2026

---

## Resumen ejecutivo

SADAR es un sistema de aprendizaje profundo que aprende el patrón normal de las
operaciones de aproximación y despegue en Madrid-Barajas (LEMD) a partir de datos de
vigilancia ADS-B y señala en tiempo real cualquier trayectoria que se aparte de ese
patrón. El sistema está concebido como **monitor de conformidad de vuelos**: no clasifica
eventos concretos, asigna a cada vuelo del espacio aéreo una puntuación interpretable de
anomalía para que un operador pueda revisar los casos que quedan fuera de la
distribución aprendida.

Se entrenan tres autoencoders secuenciales en igualdad de condiciones sobre unos 20.000
vuelos reales obtenidos de la red OpenSky (un autoencoder LSTM, un autoencoder
Transformer y un VAE-LSTM), junto con un baseline no secuencial basado en Isolation
Forest. Los cuatro detectores se evalúan sobre el mismo conjunto de test y sobre un banco
sintético con cinco tipos de inyección y varias intensidades. Se selecciona el
**VAE-LSTM** como modelo de producción: obtiene la mejor PR-AUC sobre datos reales
(0,299) y la mejor ROC-AUC media en el banco sintético (0,792). También se evalúan dos
estrategias de ensemble (consenso y más sensible) y se descartan, porque ninguna mejora
al mejor modelo individual.

El modelo seleccionado se sirve a través de una aplicación FastAPI y de un dashboard
React + Vite que combina un radar en vivo estilo torre de control, un ranking de
anomalías, una página de métricas y un simulador interactivo. Todo el sistema es
reproducible desde una única imagen Docker y está listo para desplegarse en Hugging Face
Spaces.

---

## 1. Motivación y alcance

La monitorización de trayectorias es una función central de la gestión del tráfico
aéreo. Hoy se apoya en reglas deterministas y en supervisión humana: separación,
perfiles de descenso, esperas o códigos de transpondedor de emergencia se vigilan
contra umbrales fijos. Esas reglas funcionan bien para las situaciones para las que
fueron diseñadas, pero no pueden anticipar comportamientos que sean inusuales sin ser
una violación de manual.

SADAR explora un enfoque complementario, basado en datos. En lugar de codificar reglas,
el sistema **aprende** cómo es una operación normal en LEMD a partir de una muestra
amplia de vuelos reales, y trata cada nueva trayectoria como más o menos compatible con
esa distribución. La salida es una puntuación continua de conformidad que el operador
puede ordenar, filtrar y umbralizar.

El alcance del proyecto es deliberadamente acotado:

- **Qué hace.** Detecta desviaciones respecto al patrón normal de tráfico de un único
  aeropuerto (LEMD) a partir de un conjunto fijo de variables dinámicas extraídas de las
  observaciones ADS-B.
- **Qué no hace.** Predecir eventos operativos concretos, sustituir al controlador,
  emitir recomendaciones de seguridad certificadas, ni generalizar sin reentrenamiento a
  aeropuertos con procedimientos distintos.

Esta formulación como **monitor de conformidad** es la que hace tratable el problema con
los datos disponibles: solo requiere ejemplos de vuelo normal, que son abundantes, y
deja clara la interpretación de la salida ("esta trayectoria no se parece a las que el
modelo ha visto"), sin sobrevender.

---

## 2. Datos

### 2.1 Fuente

El dataset se construye a partir de observaciones ADS-B captadas por la red OpenSky en
la zona terminal de Madrid-Barajas. Cubre aproximadamente 18 días de operaciones
muestreados entre junio de 2017 y marzo de 2020. Cada aeronave se reporta una vez cada
10 segundos, la cadencia natural de ADS-B.

| Propiedad | Valor |
|---|---|
| Trayectorias totales | ≈ 20.000 (≈ 950 a 1.200 por día) |
| Filas por día | 140.000 a 240.000 |
| Columnas | 21 |
| Puntos por vuelo (mín · mediana · máx) | 40 · 188 · 1.246 |

### 2.2 Variables

El esquema en bruto contiene información posicional (latitud, longitud, tiempo),
dinámica (altitud barométrica y geométrica, velocidad respecto al suelo, rumbo, régimen
vertical), campos derivados (distancia a pista, fase de vuelo), identidad (ICAO24,
indicativo, identificador de vuelo) y estado del transpondedor (squawk, indicador de
"en tierra", indicador de alerta, SPI y último contacto). Un campo, `operation`, está
vacío en todo el dataset y se descarta.

### 2.3 Calidad y limpieza

Un análisis de perfilado sobre el dataset completo revela varios problemas que se
gestionan en el pipeline de preprocesado:

- **Valores ausentes.** `geoaltitude` falta en el 34,9 % de las filas, `baroaltitude` en
  el 19,6 %, los campos cinemáticos (`velocity`, `heading`, `vertrate`) en torno al
  19 %. El resto se queda por debajo del 4 %.
- **Archivo duplicado.** Un día llega dos veces con nombres distintos y se deduplica por
  hash de contenido.
- **Días vacíos o erróneos.** Algunos días llegan sin filas o con errores en el
  manifest y se descartan.
- **Huecos de cobertura.** Cerca del 8 % de los vuelos en aire muestran un salto
  superior a 120 s entre observaciones consecutivas. Casi siempre son artefactos de
  cobertura del receptor, no eventos operativos, por lo que se utilizan como
  característica, jamás como etiqueta.

Tras la limpieza, el dataset contiene un conjunto estable de operaciones normales en
LEMD que abarca varias estaciones e incluye el inicio de 2020, lo que sirve además como
cambio de distribución natural para poner a prueba el split temporal.

### 2.4 Observaciones destacadas

El análisis exploratorio confirma un pequeño conjunto de casos raros que se reservan
únicamente para validación:

- Cuatro vuelos llevan un código de transpondedor de emergencia en algún punto.
- Aproximadamente un centenar de vuelos muestran un patrón claro de motor y al aire
  (aproximación seguida de ascenso sostenido).

Ninguno de estos vuelos entra en el conjunto de entrenamiento. Se utilizan como prueba
de cordura del modelo desplegado.

---

## 3. Formulación del problema

El dataset contiene muchos vuelos normales y casi ninguna anomalía etiquetada. Eso
descarta la clasificación supervisada y apunta al **aprendizaje de una sola clase**:
entrenar un modelo que capture la distribución de trayectorias normales y puntuar los
nuevos vuelos según su compatibilidad con esa distribución.

En la práctica, los tres modelos profundos son **autoencoders** entrenados solo con
vuelos normales. El objetivo de entrenamiento es reconstruir la ventana de entrada a
partir de una representación latente comprimida; en inferencia se utiliza el **error de
reconstrucción** como puntuación de conformidad. Una trayectoria parecida a los patrones
normales se reconstruye con precisión y recibe un score bajo; una trayectoria que se
aparta de esos patrones es más difícil de reconstruir y recibe un score alto.

Esta formulación encaja con el contexto operativo por tres motivos. Sólo necesita datos
normales, que son los únicos disponibles a escala. Produce una puntuación continua e
interpretable en lugar de una etiqueta dura, lo que es la salida adecuada para una
herramienta de apoyo a un operador humano. Y se comporta bien ante modos de fallo
desconocidos: cualquier trayectoria que el modelo no haya aprendido a reconstruir
emerge como anómala con independencia de qué es lo que la hace inusual.

---

## 4. Pipeline de preprocesado

El pipeline transforma los parquet en bruto en tensores de longitud fija ya
estandarizados y listos para entrenar. Cada paso está implementado como una función
pequeña y testeable bajo `src/sadar/data/`.

1. **Carga y fusión.** Se leen todos los parquet bajo `$SADAR_DATA_DIR` con pyarrow, se
   concatenan y se deduplican. Los días vacíos o erróneos se descartan.
2. **Transformación de coordenadas.** La latitud y la longitud se proyectan a un sistema
   métrico relativo a pista con `pyproj`. Así el modelo es robusto a la curvatura de las
   coordenadas y recibe metros como unidad.
3. **Codificación del rumbo.** El rumbo se codifica como `(sin(rumbo), cos(rumbo))` para
   evitar la discontinuidad entre 0 y 360 grados.
4. **Tratamiento de ausentes.** Los huecos cortos dentro de una trayectoria por lo demás
   completa se rellenan con interpolación lineal. Las trayectorias con demasiados
   ausentes se descartan.
5. **Resampleado.** Cada vuelo se resamplea a una rejilla uniforme de 10 s, que coincide
   con la cadencia nominal de ADS-B.
6. **Ventanado.** Cada trayectoria resampleada se trocea en ventanas de longitud fija.
   La longitud de ventana es un hiperparámetro definido en
   `configs/preprocessing.yaml`.
7. **Estandarización.** Cada variable se normaliza a media cero y varianza unidad. El
   escalador se ajusta **solo con el conjunto de entrenamiento** para evitar fugas, y se
   guarda como `data/processed/scaler.npz`, de manera que la misma transformación se
   aplica en inferencia.
8. **Split sin fugas.** La división se hace conjuntamente por `flight_id` y por fecha:
   ningún vuelo aparece en más de un conjunto, y el test es estrictamente posterior en
   el tiempo al entrenamiento. Los años 2017 a 2019 se usan para entrenamiento y
   validación; 2020 se reserva para test. Este split realista es lo que permite tomar
   las cifras de latencia con confianza.
9. **Limpieza del set de entrenamiento.** Los pocos vuelos con squawks de emergencia y
   los go-around sospechosos se eliminan del entrenamiento. Sólo trayectorias
   "normales" entran al modelo.

La entrada de todos los modelos profundos es una ventana de siete variables:
`[x_rel, y_rel, baroaltitude, velocity, sin_hdg, cos_hdg, vertrate]`.

---

## 5. Modelos

### 5.1 Baseline Isolation Forest

Detector de referencia, no secuencial. Cada ventana se resume por estadísticos por
variable (media, desviación, mínimo, máximo) y se alimenta a un Isolation Forest. El
baseline sirve para cuantificar cuánto añade realmente el modelado secuencial.

### 5.2 Autoencoder LSTM

Un encoder LSTM de dos capas comprime la ventana de entrada en un vector latente, y un
decoder LSTM espejado la reconstruye. Se entrena de extremo a extremo con MSE y el loop
de entrenamiento compartido descrito más abajo. Es la arquitectura caballo de batalla:
estable, rápida en inferencia y alineada con la bibliografía sobre modelado de
trayectorias ADS-B.

### 5.3 Autoencoder Transformer

El mismo esqueleto encoder-decoder, pero con atención multi-cabeza en lugar de
recurrencia. El Transformer captura dependencias a largo plazo dentro de la ventana sin
el cuello de botella secuencial de las LSTM, a cambio de más parámetros y una superficie
de hiperparámetros mayor.

### 5.4 VAE-LSTM (modelo de producción seleccionado)

Un autoencoder variacional construido sobre encoder y decoder LSTM. En lugar de producir
un único vector latente, el encoder produce una distribución posterior; el decoder
reconstruye a partir de una muestra. La función de pérdida es la ELBO habitual: MSE de
reconstrucción más un término KL que regulariza la posterior hacia una gaussiana
estándar, escalado por un factor beta. Esto aporta una interpretación probabilística al
score y se comporta mejor en el paso de selección de umbral.

### 5.5 Ensembles (evaluados y descartados)

Como parte de la comparación se evalúan dos estrategias de ensemble. Tras calibrar cada
modelo con z-score sobre las normales de validación, los tres modelos profundos se
combinan por la **media** de sus scores (consenso) o por el **máximo** (más sensible).
Ninguna combinación supera al mejor modelo individual en el test real, así que el
ensemble queda documentado y descartado. El detector final es el VAE-LSTM en solitario.

### 5.6 Infraestructura de entrenamiento compartida

Los tres modelos profundos comparten el mismo loop de entrenamiento, implementado una
sola vez en `src/sadar/models/training.py`. El loop usa:

- **Adam** con weight decay,
- planificador **ReduceLROnPlateau**,
- **early stopping** sobre la pérdida de validación,
- **MLflow** para tracking (losses, hiperparámetros, mejor checkpoint),
- **Optuna** para búsqueda de hiperparámetros, usado sobre todo en el VAE-LSTM.

Cada arquitectura tiene su propio fichero YAML bajo `configs/`. La reproducibilidad se
garantiza con semillas fijas y guardando junto al checkpoint la configuración exacta de
preprocesado utilizada.

---

## 6. Banco de anomalías sintéticas

Como las anomalías reales son escasas y carecen de etiqueta, se construye un banco
controlado sobre los vuelos normales del test. Cada escenario parte de una ventana
normal real e inyecta una perturbación conocida y etiquetada que se aplica de forma
gradual a partir de la mitad de la ventana. Se implementan cinco tipos de inyección,
cada uno en varias intensidades:

| Tipo | Descripción |
|---|---|
| Desviación de ruta | Salida lateral del corredor, parametrizada en metros. |
| Anomalía de altitud | Desplazamiento de altitud inconsistente con la fase. |
| Anomalía de velocidad | Aceleración o deceleración multiplicativa. |
| Holding | Giro de 360 ° repetido con periodo configurable. |
| Congelado de transpondedor | La señal se mantiene fija desde un punto dado. |

El banco produce etiquetas de verdad (cada inyección sabe exactamente cuándo empieza) y
por tanto permite dos métricas que los datos reales no pueden dar: **PR-AUC por tipo de
anomalía** y **latencia de detección** (mediana de segundos entre el inicio de la
inyección y el primer instante en que el score por paso cruza el umbral de alerta).

El banco sintético se utiliza únicamente para evaluación. Ninguna ventana sintética
entra jamás en el entrenamiento.

---

## 7. Evaluación

### 7.1 Conjunto de test real

Los cuatro detectores se puntúan sobre el mismo split de test. La métrica principal es
**PR-AUC**, más informativa que la ROC-AUC dado el fuerte desbalance de clases en datos
reales.

| Modelo | ROC-AUC | PR-AUC | ROC-AUC media sintética | Latencia mediana |
|---|---|---|---|---|
| Baseline Isolation Forest | 0,515 | 0,133 | 0,593 | n/d |
| Autoencoder LSTM | 0,648 | 0,260 | 0,779 | 115 s |
| Autoencoder Transformer | 0,614 | 0,227 | 0,743 | 115 s |
| **VAE-LSTM (seleccionado)** | **0,659** | **0,299** | **0,792** | 120 s |
| Ensemble (media) | 0,652 | 0,273 | 0,778 | n/d |
| Ensemble (máximo) | 0,658 | 0,278 | 0,774 | n/d |

Se desprenden tres conclusiones:

- Los modelos profundos aportan valor sustancial sobre el baseline (la PR-AUC
  aproximadamente se duplica).
- El VAE-LSTM es el mejor en conjunto: mayor PR-AUC sobre datos reales, mayor ROC-AUC
  media sintética y latencia competitiva.
- Las dos estrategias de ensemble no superan al mejor modelo individual y no se
  retienen.

### 7.2 Banco sintético (ROC-AUC por anomalía)

El VAE-LSTM es el mejor en todos los tipos de anomalía. El patrón es consistente: las
perturbaciones grandes y sostenidas son más fáciles que las sutiles, y las maniobras
estructuradas (holding, cambios fuertes de velocidad) son más fáciles que los
desplazamientos pequeños.

| Anomalía | Intensidad | Baseline | LSTM | Transformer | VAE-LSTM |
|---|---|---|---|---|---|
| Desviación de ruta | 20.000 m | 0,513 | 0,556 | 0,543 | 0,558 |
| Desviación de ruta | 40.000 m | 0,537 | 0,684 | 0,657 | 0,700 |
| Desviación de ruta | 80.000 m | 0,618 | 0,883 | 0,865 | 0,899 |
| Desviación de altitud | 300 m | 0,502 | 0,512 | 0,505 | 0,514 |
| Desviación de altitud | 800 m | 0,518 | 0,581 | 0,531 | 0,593 |
| Desviación de altitud | 1.500 m | 0,561 | 0,726 | 0,606 | 0,752 |
| Factor de velocidad | x1,6 | 0,620 | 0,901 | 0,835 | 0,927 |
| Factor de velocidad | x2,2 | 0,743 | 0,985 | 0,965 | 0,989 |
| Factor de velocidad | x0,4 | 0,673 | 0,901 | 0,884 | 0,925 |
| Holding | 240 s/giro | 0,614 | 0,972 | 0,979 | 0,978 |
| Holding | 120 s/giro | 0,634 | 0,975 | 0,985 | 0,984 |
| Congelado transpondedor | fijo | 0,582 | 0,668 | 0,558 | 0,686 |

### 7.3 Curvas y matrices de confusión

Para cada detector se generan curvas precision-recall y ROC, junto con una matriz de
confusión tomada al umbral del percentil 99 de validación. Las figuras se guardan como
PNG en `reports/figures/` y los arrays subyacentes se almacenan dentro de
`reports/model_comparison.json`, de modo que el dashboard puede renderizarlas sin
recalcular.

---

## 8. Inferencia y servicio

El VAE-LSTM seleccionado se sirve mediante una aplicación FastAPI compacta
(`src/sadar/serve/app.py`). En el arranque el servicio carga:

- el checkpoint del modelo de producción,
- el escalador ajustado con los datos de entrenamiento,
- el informe de métricas precomputado,
- y una muestra curada de trayectorias para que el dashboard pueda funcionar sin los
  datos brutos.

La superficie HTTP es deliberadamente reducida. El dashboard se comunica con el backend
a través de cinco endpoints:

| Endpoint | Función |
|---|---|
| `GET /api/health` | Sonda de vida (también devuelve un identificador de build). |
| `GET /api/scene` | Escena de radar actual: aeronaves, scores, umbral. |
| `GET /api/flights` | Lista ordenada de vuelos monitorizados con metadatos. |
| `GET /api/metrics` | Métricas comparativas y curvas precomputadas. |
| `POST /api/simulate` | Inyecta una anomalía en un vuelo y devuelve la respuesta. |

En producción, el mismo proceso FastAPI también sirve la SPA de React ya compilada (la
SPA se monta en `/`, la API queda bajo `/api`). Todo el sistema corre en un único
contenedor.

---

## 9. Frontend

El dashboard es una aplicación React 18 + Vite + TypeScript diseñada para que parezca y
se sienta como una consola de torre de control: fondo oscuro, bloques de datos en
monoespaciada, criticidad codificada por color (verde y cian para normal, ámbar y rojo
para alerta). Expone cuatro pantallas:

1. **Consola torre.** Un radar en directo centrado en LEMD con un panel lateral que
   lista todos los vuelos monitorizados y su estado actual. Las aeronaves en seguimiento
   o en alerta se diferencian por color.
2. **Simulador.** El corazón interactivo de la demo. El operador elige un vuelo, un tipo
   de anomalía, una intensidad y un instante de inicio, e inyecta la perturbación en
   tiempo real. El score de la derecha reacciona a la inyección y la latencia de
   detección se reporta en segundos.
3. **Métricas.** El informe comparativo completo: curvas ROC y PR, desglose de PR-AUC
   por anomalía, justificación de la selección del modelo y resumen del detector final.
4. **Presentación.** Un recorrido guiado utilizado para presentaciones en vivo del
   proyecto.

La aplicación está internacionalizada en inglés y español mediante un pequeño contexto
de traducción.

---

## 10. Reproducibilidad y despliegue

El proyecto está diseñado para que el pipeline completo (preprocesado, entrenamiento,
evaluación, servicio) pueda reproducirse desde cero.

- **Entorno Python.** Fijado con `uv` y `pyproject.toml`; un único comando
  (`uv sync`) reconstruye el entorno.
- **Entorno de frontend.** Gestionado con `pnpm` y `pnpm-lock.yaml`.
- **Configuración.** Todos los parámetros (rutas, longitud de ventana, dimensiones del
  modelo, plan de entrenamiento, umbrales de evaluación) viven en ficheros YAML bajo
  `configs/`. El código no lee rutas literales.
- **Semillas.** Una única semilla se fija al inicio de cada entry point y se propaga a
  NumPy, scikit-learn y PyTorch.
- **Tracking.** MLflow registra todas las ejecuciones en local; Optuna guarda su
  historial de búsqueda junto a los hiperparámetros resultantes.
- **Containerización.** Un único Dockerfile construye una imagen autocontenida que
  incluye el modelo entrenado, el escalador, los informes precomputados y la SPA. La
  misma imagen se utiliza en local (`docker compose up`) y en producción (Hugging Face
  Spaces).

Un `Makefile` expone el flujo como targets con nombre: `make install`,
`make preprocess`, `make train-lstm`, `make train-transformer`, `make train-vae`,
`make tune-vae`, `make compare`, `make serve`, `make dev`, `make docker-up`.

---

## 11. Limitaciones y trabajo futuro

### 11.1 Limitaciones

- **Alcance.** El dataset cubre un solo aeropuerto (LEMD) y un periodo acotado. No se
  espera que el modelo transfiera a otras terminales sin reentrenamiento.
- **Sin planes de vuelo.** La "ruta prevista" se aproxima como "patrón normal
  aprendido". Un vuelo que se aparte de su plan de vuelo pero permanezca dentro del
  patrón normal del tráfico podría no ser marcado.
- **Huecos de cobertura.** Los huecos del transpondedor en ADS-B están dominados por la
  cobertura del receptor. Se usan como característica de entrada, no como etiqueta.
- **Desbalance de clases.** Los eventos anómalos reales son extremadamente raros. Las
  cifras de PR-AUC sobre el test real se calculan con un conjunto positivo pequeño; el
  banco sintético es el referente cuantitativo principal.

### 11.2 Trabajo futuro

- Ampliar el dataset a más aeropuertos y validar la generalización entre terminales.
- Incorporar los planes de vuelo, cuando estén disponibles, como señal auxiliar.
- Combinar el score de conformidad con un clasificador posterior entrenado sobre
  eventos operativos etiquetados (motor y al aire, aproximaciones frustradas, holding)
  para producir alertas más interpretables.
- Explorar ensembles probabilísticos que ponderen cada modelo por su verosimilitud
  calibrada en lugar de hacerlo con un reductor fijo.

---

## 12. Conclusiones

SADAR muestra que un enfoque de aprendizaje profundo de una sola clase es una base
viable para la monitorización de conformidad de trayectorias en un aeropuerto
importante. Tres autoencoders secuenciales entrenados con los mismos datos y evaluados
sobre el mismo banco arrojan una elección de modelo clara y defendible: el
**VAE-LSTM** combina la mejor PR-AUC sobre datos reales, el mejor desempeño medio en el
banco sintético y una latencia mediana de detección competitiva, mejorando con holgura
a un baseline no secuencial. El sistema es completamente reproducible, se empaqueta en
un único contenedor y se sirve a través de un dashboard construido en torno a un caso
de uso real.

El resultado no es un producto cerrado, pero sí un prototipo sólido de cómo la
monitorización de conformidad basada en datos puede complementar las herramientas
basadas en reglas que usan hoy los operadores del tráfico aéreo.

---

**Repositorio.** Smart Anomaly Detection for Aviation Routes (SADAR).
Madrid-Barajas (LEMD) · 2026.
