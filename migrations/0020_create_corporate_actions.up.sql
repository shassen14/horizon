CREATE TABLE corporate_actions (
  asset_id INTEGER NOT NULL REFERENCES assets (id),
  ex_date DATE NOT NULL,
  action_type TEXT NOT NULL CHECK (action_type IN ('split', 'dividend')),
  -- split: new shares per old share (4.0 = a 4:1 split). NULL for dividends.
  split_ratio NUMERIC(18, 9),
  -- dividend: cash per share. NULL for splits.
  cash_amount NUMERIC(18, 9),
  source TEXT NOT NULL,
  ingested_at TIMESTAMPTZ NOT NULL DEFAULT now (),
  PRIMARY KEY (asset_id, ex_date, action_type),
  -- Each action kind carries exactly its own payload, nothing else.
  CONSTRAINT corporate_actions_payload CHECK (
    (
      action_type = 'split'
      AND split_ratio IS NOT NULL
      AND cash_amount IS NULL
    )
    OR (
      action_type = 'dividend'
      AND cash_amount IS NOT NULL
      AND split_ratio IS NULL
    )
  )
);
