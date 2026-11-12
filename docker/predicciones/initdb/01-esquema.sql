-- Esquema del registro de predicciones (Hito 7b).
--
-- Lo corre la imagen oficial de Postgres UNA sola vez: al primer arranque, cuando el
-- volumen está vacío. Cambiar este archivo después NO modifica una base que ya existe;
-- eso es trabajo de una migración (pendiente: Alembic, cuando el esquema cambie).
--
-- Tres roles, mínimo privilegio:
--   dueno          crea el esquema (POSTGRES_USER). Nadie más lo usa.
--   api_escritura  la API: SÓLO INSERT. No puede leer, borrar ni cambiar lo que escribió.
--   lector         Grafana y el análisis de drift (7c): SÓLO SELECT.
-- Si alguien roba la contraseña de la API, puede agregar filas falsas, pero no leer las
-- predicciones de nadie ni borrar el historial.
--
-- Contraseñas: llegan como variables de entorno (las genera tasks.py en .env). `\getenv`
-- es un comando de psql (≥ 15) que lee una variable de entorno a una variable de psql.

\getenv pw_api PREDICCIONES_API_PASSWORD
\getenv pw_lector PREDICCIONES_LECTOR_PASSWORD

CREATE TABLE predicciones (
    id           uuid PRIMARY KEY,          -- lo genera la API; viaja en X-Prediccion-Id
    ts           timestamptz NOT NULL,      -- cuándo se respondió (UTC)
    segmento     text NOT NULL,             -- ocupados | no_ocupados
    modelo       text NOT NULL,             -- ingreso_ocupados
    version      text NOT NULL,             -- versión del registry ("4")
    commit_api   text NOT NULL,             -- qué código respondió
    pod          text NOT NULL,             -- qué réplica respondió
    -- Las features EXACTAS que vio el modelo. jsonb y no una columna por feature: los
    -- dos segmentos usan features distintas y un reentrenamiento puede cambiarlas; con
    -- columnas, cada modelo nuevo exigiría una migración. Costo: consultar es un poco
    -- más verboso (features->>'cve_entidad').
    features     jsonb NOT NULL,
    inferior     double precision NOT NULL,
    mediana      double precision NOT NULL,
    superior     double precision NOT NULL,
    latencia_ms  real NOT NULL              -- sólo el modelo: sin red ni validación
);
-- El drift compara ventanas de tiempo por segmento: ése es el acceso que se indexa.
CREATE INDEX predicciones_segmento_ts ON predicciones (segmento, ts);

-- Nadie entra por omisión; cada rol recibe sólo lo suyo.
REVOKE ALL ON DATABASE predicciones FROM PUBLIC;

CREATE ROLE api_escritura LOGIN PASSWORD :'pw_api';
GRANT CONNECT ON DATABASE predicciones TO api_escritura;
GRANT USAGE ON SCHEMA public TO api_escritura;
GRANT INSERT ON predicciones TO api_escritura;

CREATE ROLE lector LOGIN PASSWORD :'pw_lector';
GRANT CONNECT ON DATABASE predicciones TO lector;
GRANT USAGE ON SCHEMA public TO lector;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO lector;
-- Las tablas que se creen después (resultados de drift, 7c) también serán legibles.
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO lector;
