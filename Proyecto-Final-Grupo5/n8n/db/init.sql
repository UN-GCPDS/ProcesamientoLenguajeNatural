\set ON_ERROR_STOP on
\getenv reader_password JALECO_READER_PASSWORD

BEGIN;

CREATE SCHEMA jaleco;

CREATE TABLE jaleco.pedidos (
    order_id text PRIMARY KEY CHECK (order_id ~ '^JS-[0-9]{3}$'),
    sku text NOT NULL,
    product_name text NOT NULL,
    channel text NOT NULL CHECK (channel IN ('online', 'tienda')),
    delivered_on date NOT NULL,
    warranty_months smallint NOT NULL CHECK (warranty_months >= 0),
    requires_serial boolean NOT NULL
);

CREATE TABLE jaleco.politicas (
    policy_id text NOT NULL,
    version integer NOT NULL CHECK (version > 0),
    category text NOT NULL CHECK (category IN ('retracto', 'cambio_talla', 'defecto', 'pedido_incompleto')),
    effective_from date NOT NULL,
    rule jsonb NOT NULL CHECK (jsonb_typeof(rule) = 'object'),
    PRIMARY KEY (policy_id, version),
    UNIQUE (category, effective_from)
);

CREATE TABLE jaleco.festivos (
    holiday_date date PRIMARY KEY,
    holiday_name text NOT NULL
);

INSERT INTO jaleco.pedidos
    (order_id, sku, product_name, channel, delivered_on, warranty_months, requires_serial)
VALUES
    ('JS-101', 'BAL-01', 'Balón de entrenamiento', 'online', '2026-11-13', 12, false),
    ('JS-102', 'CRO-02', 'Cronómetro deportivo', 'online', '2026-11-12', 12, true),
    ('JS-103', 'CAM-M', 'Camiseta de entrenamiento M', 'tienda', '2026-11-14', 6, false),
    ('JS-104', 'KIT-04', 'Kit de bandas elásticas', 'online', '2026-11-17', 12, false);

INSERT INTO jaleco.politicas (policy_id, version, category, effective_from, rule)
VALUES
    ('RETRACTO_CO', 1, 'retracto', '2026-01-01',
     '{"mode":"deadline","channel":"online","startField":"deliveredOn","window":{"value":5,"unit":"business_days"},"source":"Ley 1480 de 2011, artículo 47"}'),
    ('CAMBIO_JALECO', 1, 'cambio_talla', '2026-01-01',
     '{"mode":"deadline","startField":"deliveredOn","window":{"value":30,"unit":"calendar_days"},"source":"Política comercial ficticia de Jaleco"}'),
    ('GARANTIA_JALECO', 1, 'defecto', '2026-01-01',
     '{"mode":"order_warranty","startField":"deliveredOn","warrantyField":"warrantyMonths","serialField":"requiresSerial"}'),
    ('INCOMPLETO_JALECO', 1, 'pedido_incompleto', '2026-01-01',
     '{"mode":"human_review"}');

INSERT INTO jaleco.festivos (holiday_date, holiday_name)
VALUES ('2026-11-16', 'Independencia de Cartagena');

CREATE ROLE jaleco_reader LOGIN PASSWORD :'reader_password';
GRANT CONNECT ON DATABASE jaleco TO jaleco_reader;
GRANT USAGE ON SCHEMA jaleco TO jaleco_reader;
GRANT SELECT ON jaleco.pedidos, jaleco.politicas, jaleco.festivos TO jaleco_reader;

COMMIT;
