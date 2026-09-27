-- Phase 2: conversation state, provenance, and exactly-once inbound handling.

ALTER TABLE inbound_events ADD COLUMN response_text TEXT;
ALTER TABLE inbound_events ADD COLUMN completed_at TEXT;

ALTER TABLE meal_revisions ADD COLUMN origin TEXT NOT NULL DEFAULT 'text';
ALTER TABLE meal_revisions ADD COLUMN model TEXT;

-- One inbound event may create at most one logical meal, however many times
-- a webhook or CLI retry redelivers it.
CREATE UNIQUE INDEX IF NOT EXISTS idx_one_meal_per_source_event
    ON meals(source_event_id) WHERE source_event_id IS NOT NULL;
