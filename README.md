# ACI606 - Cloud Fleet Challenge

Proyecto para el **Ejercicio 4** de ACI606 Cloud Computing.

## URL del Servicio Desplegado (Entregable Final)

El servicio está completamente desplegado y es consumible remotamente en la plataforma **Render**, utilizando una base de datos gestionada en **Neon**.

- **Dashboard del Operador:** [https://aci606-cloud-fleet-challenge-bi49.onrender.com/dashboard](https://aci606-cloud-fleet-challenge-bi49.onrender.com/dashboard)
- **Documentación Interactiva (Swagger):** [https://aci606-cloud-fleet-challenge-bi49.onrender.com/docs](https://aci606-cloud-fleet-challenge-bi49.onrender.com/docs)
- **Métricas de Observabilidad:** [https://aci606-cloud-fleet-challenge-bi49.onrender.com/metrics](https://aci606-cloud-fleet-challenge-bi49.onrender.com/metrics)
- **Estado del Servicio (Health):** [https://aci606-cloud-fleet-challenge-bi49.onrender.com/health](https://aci606-cloud-fleet-challenge-bi49.onrender.com/health)

*(Nota: Al usar el plan gratuito de Render, el servicio se suspende tras 15 minutos de inactividad. La primera petición puede tardar hasta 50 segundos en "despertar" el servicio. Se recomienda abrir `/health` primero).*

## Arquitectura de la Solución

1. **Cliente (Simulador):** Genera telemetría y envía peticiones HTTP a la API.
2. **API (FastAPI en Render):** Recibe los datos, ejecuta el modelo de Machine Learning y expone los endpoints.
3. **Base de Datos (Neon PostgreSQL):** Almacena de forma persistente todo el historial de telemetría.
4. **Dashboard:** Interfaz web (HTML/JS) que consume la API para mostrar el estado de la flota en tiempo real.

## Requisitos Implementados (Ejercicio 4)

- **1. Ingesta persistente:** Reemplazo de RAM por PostgreSQL en Neon. Los datos sobreviven a reinicios.
- **2. Entrenamiento de ML:** Generación de dataset, separación con `GroupShuffleSplit` por `fleet_id`, comparación de Regresión Logística vs Random Forest, y guardado en `models/model.joblib`.
- **3. Servicio de inferencia:** Endpoint `POST /predict` que devuelve el contrato exacto (status, confianza, versión y latencia).
- **4. Dashboard:** Panel visual que muestra robots conectados, batería, predicciones en colores y métricas del sistema.
- **5. Observabilidad:** Endpoint `GET /metrics` que cuenta peticiones, errores, latencia promedio y uptime.
- **6. Trazabilidad Git:** Ramas de funcionalidad, commits significativos y merge a `main`.
- **7. Ejecución remota:** Desplegado en Render con URL pública.

## Modelo de Machine Learning

El entrenamiento se realizó evaluando dos clasificadores con un Pipeline reproducible (StandardScaler + Clasificador).

### Evidencia y Resultados
- **Distribución de clases:** CRITICAL (2153), WARNING (534), NORMAL (313).
- **Manejo de desbalance:** Se utilizó `class_weight='balanced'` para dar mayor peso a las clases minoritarias (NORMAL y WARNING) y evitar que el modelo se sesgue hacia CRITICAL.
- **Separación Train/Test:** Se implementó `GroupShuffleSplit` por `fleet_id`. Al no incluir el dataset generado esta columna, el script cuenta con un *fallback* seguro a una separación estratificada (`train_test_split` con `stratify=y`) para mantener la proporción de clases.

### Comparación de Modelos
| Modelo | F1 Macro | Accuracy |
| :--- | :--- | :--- |
| Regresión Logística | 0.9155 | 0.94 |
| **Random Forest (Ganador)** | **0.9440** | **0.96** |

**Justificación técnica:** Se seleccionó **Random Forest** por obtener el mayor F1 Macro (0.9440). Esta métrica fue elegida como principal porque el dataset está fuertemente desbalanceado (la clase CRITICAL domina). El F1 Macro castiga al modelo si ignora las clases minoritarias, garantizando que se detecten tanto las fallas críticas como las advertencias tempranas.

---

## Cómo ejecutar el proyecto localmente

### A. Crear entorno e instalar dependencias
    python -m venv .venv
    source .venv/bin/activate  # En Windows usa: .venv\Scripts\activate
    pip install -r requirements.txt

### B. Configurar Variables de Entorno
Crea un archivo llamado **`.env`** en la raíz del proyecto (al mismo nivel que `requirements.txt`) con el siguiente contenido (reemplaza con tus credenciales reales de Neon):

    DATABASE_URL=postgresql+psycopg://neondb_owner:TU_CONTRASEÑA@ep-flat-water-b5qxxesc-pooler.c-7.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require
    MODEL_PATH=models/model.joblib

*(Nota: El archivo `.env` está incluido en el archivo `.gitignore` para evitar que las contraseñas se filtren en GitHub).*

### C. Generar datos y entrenar el modelo

    python simulator/generate_dataset.py --samples 3000
    python ml/train_model.py

### D. Iniciar la API

    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

### E. Iniciar el simulador de flota (en otra terminal)

    python simulator/fleet_simulator.py --api http://127.0.0.1:8000 --robots 5

---

## Procedimiento de Despliegue en Render (PaaS)

El despliegue se realizó en la plataforma **Render**, conectando el repositorio de GitHub para lograr un flujo de Integración y Despliegue Continuo (CI/CD).

1. **Conexión del Repositorio:** Se conectó la rama `main` de GitHub a un *Web Service* en Render.
2. **Build Command:** `pip install -r requirements.txt` (Instala las dependencias en el contenedor de Render).
3. **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT` (Arranca el servidor escuchando en el puerto dinámico asignado por Render).
4. **Variables de Entorno en Render:**
   - `DATABASE_URL`: Cadena de conexión a Neon.
   - `MODEL_PATH`: `models/model.joblib`.

## Integrantes del Grupo
- Roberto Abarzúa
- Ricardo Gutierrez
- José Llancamil