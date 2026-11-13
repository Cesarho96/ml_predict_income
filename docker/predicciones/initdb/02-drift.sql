-- Migración 02 (Hito 7c): resultados de drift y el rol que los calcula.
--
-- Corre en DOS situaciones, y debe funcionar en ambas:
--   · base nueva: la imagen de Postgres ejecuta 01 y luego 02, en orden;
--   · base que ya existe (la tuya, con predicciones de 7b): `python tasks.py migrar`.
-- Por eso es IDEMPOTENTE: correrla dos veces deja lo mismo que correrla una
-- (IF NOT EXISTS, y el rol sólo se crea si no existe). Es la versión artesanal de lo que
-- hacen Alembic o Flyway: ellos además llevan la cuenta de qué migración ya corrió.

\getenv pw_monitor PREDICCIONES_MONITOR_PASSWORD

-- Una fila por (corrida, segmento, feature). Se acumula: el historial de PSI ES el
-- monitoreo; Grafana grafica su evolución.
CREATE TABLE IF NOT EXISTS drift (
    corrida         text NOT NULL,            -- run_id de Airflow, o manual-xxxx
    calculado       timestamptz NOT NULL,
    ventana_inicio  timestamptz NOT NULL,
    ventana_fin     timestamptz NOT NULL,
    segmento        text NOT NULL,
    feature         text NOT NULL,
    tipo            text NOT NULL,            -- categorica | numerica
    psi             double precision,         -- NULL si no hubo predicciones
    n_actual        integer NOT NULL,
    n_minimo        integer NOT NULL,         -- debajo de esto el PSI es ruido
    nivel           text NOT NULL,            -- estable | vigilar | actuar | insuficiente
    detalle         jsonb NOT NULL,           -- por bin: p_ref, p_actual, aporte
    PRIMARY KEY (corrida, segmento, feature)  -- reintentar la misma corrida no duplica
);
CREATE INDEX IF NOT EXISTS drift_calculado ON drift (calculado);

-- `monitor`: lee predicciones y escribe resultados. Ni borra ni toca nada más.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'monitor') THEN
        CREATE ROLE monitor LOGIN;
    END IF;
END
$$;
-- Fuera del bloque DO: las variables de psql (:'pw_monitor') no entran en él.
ALTER ROLE monitor PASSWORD :'pw_monitor';
GRANT CONNECT ON DATABASE predicciones TO monitor;
GRANT USAGE ON SCHEMA public TO monitor;
GRANT SELECT ON predicciones TO monitor;
GRANT SELECT, INSERT, UPDATE ON drift TO monitor;

-- Grafana (lector) lee los resultados.
GRANT SELECT ON drift TO lector;
