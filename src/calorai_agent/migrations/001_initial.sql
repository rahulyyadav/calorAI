PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO schema_migrations(version) VALUES (1);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    timezone TEXT NOT NULL DEFAULT 'UTC',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    channel TEXT NOT NULL,
    external_id TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(channel, external_id)
);

CREATE TABLE IF NOT EXISTS inbound_events (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    external_id TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    raw_text TEXT,
    received_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meals (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source_event_id TEXT REFERENCES inbound_events(id) ON DELETE SET NULL,
    meal_type TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_meals_user_occurred
    ON meals(user_id, occurred_at) WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS meal_revisions (
    id TEXT PRIMARY KEY,
    meal_id TEXT NOT NULL REFERENCES meals(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    source_text TEXT NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL,
    superseded_at TEXT,
    UNIQUE(meal_id, revision_number)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_revision
    ON meal_revisions(meal_id) WHERE superseded_at IS NULL;

CREATE TABLE IF NOT EXISTS meal_items (
    id TEXT PRIMARY KEY,
    revision_id TEXT NOT NULL REFERENCES meal_revisions(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    quantity TEXT NOT NULL,
    unit TEXT NOT NULL,
    calories TEXT NOT NULL,
    protein_g TEXT NOT NULL,
    carbs_g TEXT NOT NULL,
    fat_g TEXT NOT NULL,
    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1)
);

CREATE INDEX IF NOT EXISTS idx_meal_items_revision ON meal_items(revision_id);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    memory_type TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
    source_event_id TEXT REFERENCES inbound_events(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    superseded_at TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_active_memory_key
    ON memories(user_id, memory_type, key) WHERE superseded_at IS NULL;

