-- Phase 2: record what each inbound message changed, so a message redelivered after a
-- crash between the mutation and the reply can still be answered from the database.

ALTER TABLE inbound_events ADD COLUMN result_kind TEXT;
ALTER TABLE inbound_events ADD COLUMN result_meal_id TEXT;
