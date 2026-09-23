CREATE TABLE IF NOT EXISTS cs_users (
  user_id VARCHAR(128) PRIMARY KEY, tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
  auth_subject VARCHAR(255) NULL, status VARCHAR(32) NOT NULL,
  display_name VARCHAR(128) NOT NULL, email_cipher VARBINARY(1024) NULL,
  email_hash BINARY(32) NULL, phone_cipher VARBINARY(512) NULL, phone_hash BINARY(32) NULL,
  version BIGINT UNSIGNED NOT NULL DEFAULT 1, created_at DATETIME(6) NOT NULL,
  updated_at DATETIME(6) NOT NULL, deleted_at DATETIME(6) NULL,
  INDEX idx_cs_users_tenant_status (tenant_id,status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS cs_sessions (
  session_id VARCHAR(128) PRIMARY KEY, user_id VARCHAR(128) NOT NULL,
  status VARCHAR(32) NOT NULL DEFAULT 'active', version BIGINT UNSIGNED NOT NULL DEFAULT 1,
  created_at DATETIME(6) NOT NULL, updated_at DATETIME(6) NOT NULL,
  INDEX idx_session_owner (user_id,status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS cs_user_addresses (
  address_id CHAR(26) PRIMARY KEY, user_id VARCHAR(128) NOT NULL, label VARCHAR(64) NULL,
  recipient_cipher VARBINARY(512) NOT NULL, phone_cipher VARBINARY(512) NOT NULL,
  province VARCHAR(64) NOT NULL, city VARCHAR(64) NOT NULL, district VARCHAR(64) NOT NULL,
  detail_cipher VARBINARY(2048) NOT NULL, postal_code VARCHAR(20) NULL,
  is_default BOOLEAN NOT NULL DEFAULT FALSE, status VARCHAR(32) NOT NULL DEFAULT 'active',
  version BIGINT UNSIGNED NOT NULL DEFAULT 1, created_at DATETIME(6) NOT NULL, updated_at DATETIME(6) NOT NULL,
  INDEX idx_addresses_owner (user_id,status,is_default)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS cs_payment_methods (
  payment_method_id CHAR(26) PRIMARY KEY, user_id VARCHAR(128) NOT NULL,
  type VARCHAR(32) NOT NULL, provider_token_cipher VARBINARY(2048) NULL, last4 CHAR(4) NULL,
  gift_card_balance DECIMAL(12,2) NULL, currency CHAR(3) NOT NULL DEFAULT 'CNY',
  status VARCHAR(32) NOT NULL DEFAULT 'active', version BIGINT UNSIGNED NOT NULL DEFAULT 1,
  created_at DATETIME(6) NOT NULL, updated_at DATETIME(6) NOT NULL,
  INDEX idx_payment_owner (user_id,status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS tc_accounts (
  account_id CHAR(26) PRIMARY KEY, user_id VARCHAR(128) NOT NULL, account_no VARCHAR(64) NOT NULL UNIQUE,
  status VARCHAR(32) NOT NULL, version BIGINT UNSIGNED NOT NULL DEFAULT 1,
  created_at DATETIME(6) NOT NULL, updated_at DATETIME(6) NOT NULL,
  INDEX idx_tc_account_owner (user_id,status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS tc_wallets (
  wallet_id CHAR(26) PRIMARY KEY, account_id CHAR(26) NOT NULL UNIQUE,
  available_balance DECIMAL(12,2) NOT NULL, currency CHAR(3) NOT NULL DEFAULT 'CNY',
  version BIGINT UNSIGNED NOT NULL DEFAULT 1, updated_at DATETIME(6) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS tc_wallet_ledger (
  ledger_id CHAR(26) PRIMARY KEY, wallet_id CHAR(26) NOT NULL, action_id CHAR(26) NOT NULL UNIQUE,
  direction VARCHAR(16) NOT NULL, amount DECIMAL(12,2) NOT NULL,
  balance_before DECIMAL(12,2) NOT NULL, balance_after DECIMAL(12,2) NOT NULL,
  entry_type VARCHAR(32) NOT NULL, reference_type VARCHAR(32) NULL, reference_id CHAR(26) NULL,
  created_at DATETIME(6) NOT NULL, INDEX idx_wallet_ledger (wallet_id,created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS tc_plans (
  plan_id CHAR(26) PRIMARY KEY, plan_code VARCHAR(64) NOT NULL, version_no INT UNSIGNED NOT NULL,
  name VARCHAR(255) NOT NULL, description TEXT NULL, monthly_price DECIMAL(12,2) NOT NULL,
  currency CHAR(3) NOT NULL DEFAULT 'CNY', data_limit_mb BIGINT NOT NULL,
  data_unlimited BOOLEAN NOT NULL DEFAULT FALSE, included_voice_minutes INT UNSIGNED NOT NULL,
  voice_unlimited BOOLEAN NOT NULL DEFAULT FALSE, voice_overage_price_per_minute DECIMAL(12,4) NOT NULL,
  refuel_price_per_gb DECIMAL(12,2) NOT NULL, max_refuel_mb_per_cycle BIGINT NOT NULL,
  roaming_supported BOOLEAN NOT NULL DEFAULT FALSE, status VARCHAR(32) NOT NULL,
  effective_from DATETIME(6) NULL, effective_to DATETIME(6) NULL,
  created_at DATETIME(6) NOT NULL, updated_at DATETIME(6) NOT NULL,
  UNIQUE KEY uk_plan_version (plan_code,version_no), INDEX idx_plan_active (status,effective_from,effective_to)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS tc_lines (
  line_id CHAR(26) PRIMARY KEY, account_id CHAR(26) NOT NULL, msisdn_cipher VARBINARY(512) NOT NULL,
  msisdn_hash BINARY(32) NOT NULL UNIQUE, status VARCHAR(32) NOT NULL, current_plan_id CHAR(26) NOT NULL,
  roaming_enabled BOOLEAN NOT NULL DEFAULT FALSE, contract_end_at DATETIME(6) NULL,
  last_plan_change_at DATETIME(6) NULL, version BIGINT UNSIGNED NOT NULL DEFAULT 1,
  created_at DATETIME(6) NOT NULL, updated_at DATETIME(6) NOT NULL,
  INDEX idx_line_account_status (account_id,status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS tc_usage_cycles (
  usage_id CHAR(26) PRIMARY KEY, line_id CHAR(26) NOT NULL, cycle_start DATETIME(6) NOT NULL,
  cycle_end DATETIME(6) NOT NULL, included_data_mb BIGINT NOT NULL DEFAULT 0,
  used_data_mb BIGINT NOT NULL DEFAULT 0, refueled_data_mb BIGINT NOT NULL DEFAULT 0,
  included_voice_minutes INT UNSIGNED NOT NULL DEFAULT 0, used_voice_minutes INT UNSIGNED NOT NULL DEFAULT 0,
  outgoing_call_count INT UNSIGNED NOT NULL DEFAULT 0, refuel_count INT UNSIGNED NOT NULL DEFAULT 0,
  version BIGINT UNSIGNED NOT NULL DEFAULT 1, updated_at DATETIME(6) NOT NULL,
  UNIQUE KEY uk_usage_cycle (line_id,cycle_start), INDEX idx_usage_line_end (line_id,cycle_end)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS tc_data_addons (
  addon_id CHAR(26) PRIMARY KEY, line_id CHAR(26) NOT NULL, usage_id CHAR(26) NULL,
  amount_mb BIGINT NOT NULL, charge_amount DECIMAL(12,2) NOT NULL, currency CHAR(3) NOT NULL,
  action_id CHAR(26) NOT NULL UNIQUE, status VARCHAR(32) NOT NULL, created_at DATETIME(6) NOT NULL,
  INDEX idx_addon_line (line_id,created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS tc_plan_change_history (
  change_id CHAR(26) PRIMARY KEY, line_id CHAR(26) NOT NULL, old_plan_id CHAR(26) NOT NULL,
  new_plan_id CHAR(26) NOT NULL, action_id CHAR(26) NOT NULL UNIQUE, status VARCHAR(32) NOT NULL,
  created_at DATETIME(6) NOT NULL, INDEX idx_plan_change_line (line_id,created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS tc_roaming_history (
  roaming_change_id CHAR(26) PRIMARY KEY, line_id CHAR(26) NOT NULL, enabled BOOLEAN NOT NULL,
  action_id CHAR(26) NOT NULL UNIQUE, created_at DATETIME(6) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS rt_products (
  product_id CHAR(26) PRIMARY KEY, product_code VARCHAR(64) NOT NULL UNIQUE,
  name VARCHAR(255) NOT NULL, category VARCHAR(128) NOT NULL, description TEXT NULL,
  status VARCHAR(32) NOT NULL, version BIGINT UNSIGNED NOT NULL DEFAULT 1,
  created_at DATETIME(6) NOT NULL, updated_at DATETIME(6) NOT NULL,
  FULLTEXT KEY ft_product (name,description)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS rt_product_variants (
  variant_id CHAR(26) PRIMARY KEY, product_id CHAR(26) NOT NULL, sku VARCHAR(128) NOT NULL UNIQUE,
  attributes_json JSON NOT NULL, price DECIMAL(12,2) NOT NULL, currency CHAR(3) NOT NULL,
  status VARCHAR(32) NOT NULL, version BIGINT UNSIGNED NOT NULL DEFAULT 1,
  created_at DATETIME(6) NOT NULL, updated_at DATETIME(6) NOT NULL,
  INDEX idx_variant_product (product_id,status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS rt_inventory (
  variant_id CHAR(26) PRIMARY KEY, available_qty INT UNSIGNED NOT NULL,
  reserved_qty INT UNSIGNED NOT NULL DEFAULT 0, version BIGINT UNSIGNED NOT NULL DEFAULT 1,
  updated_at DATETIME(6) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS rt_orders (
  order_id CHAR(26) PRIMARY KEY, order_no VARCHAR(64) NOT NULL UNIQUE, user_id VARCHAR(128) NOT NULL,
  status VARCHAR(32) NOT NULL, item_total DECIMAL(12,2) NOT NULL, shipping_fee DECIMAL(12,2) NOT NULL DEFAULT 0,
  discount_total DECIMAL(12,2) NOT NULL DEFAULT 0, grand_total DECIMAL(12,2) NOT NULL,
  currency CHAR(3) NOT NULL, cancel_reason VARCHAR(64) NULL, items_modified BOOLEAN NOT NULL DEFAULT FALSE,
  version BIGINT UNSIGNED NOT NULL DEFAULT 1, placed_at DATETIME(6) NOT NULL,
  created_at DATETIME(6) NOT NULL, updated_at DATETIME(6) NOT NULL,
  INDEX idx_order_owner_status (user_id,status,placed_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS rt_order_items (
  order_item_id CHAR(26) PRIMARY KEY, order_id CHAR(26) NOT NULL, product_id CHAR(26) NOT NULL,
  variant_id CHAR(26) NOT NULL, sku_snapshot VARCHAR(128) NOT NULL, name_snapshot VARCHAR(255) NOT NULL,
  attributes_snapshot_json JSON NOT NULL, unit_price DECIMAL(12,2) NOT NULL, quantity INT UNSIGNED NOT NULL,
  returned_qty INT UNSIGNED NOT NULL DEFAULT 0, exchanged_qty INT UNSIGNED NOT NULL DEFAULT 0,
  item_status VARCHAR(32) NOT NULL DEFAULT 'active', version BIGINT UNSIGNED NOT NULL DEFAULT 1,
  INDEX idx_order_item_order (order_id,item_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS rt_order_addresses (
  order_address_id CHAR(26) PRIMARY KEY, order_id CHAR(26) NOT NULL,
  address_type VARCHAR(16) NOT NULL, source_address_id CHAR(26) NULL,
  address_snapshot_json JSON NOT NULL, version BIGINT UNSIGNED NOT NULL DEFAULT 1,
  created_at DATETIME(6) NOT NULL, updated_at DATETIME(6) NOT NULL,
  UNIQUE KEY uk_order_address_type (order_id,address_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS rt_order_payments (
  payment_id CHAR(26) PRIMARY KEY, order_id CHAR(26) NOT NULL,
  payment_method_id CHAR(26) NULL, amount DECIMAL(12,2) NOT NULL,
  currency CHAR(3) NOT NULL, status VARCHAR(32) NOT NULL,
  version BIGINT UNSIGNED NOT NULL DEFAULT 1, created_at DATETIME(6) NOT NULL, updated_at DATETIME(6) NOT NULL,
  INDEX idx_order_payment (order_id,status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS rt_order_changes (
  change_id CHAR(26) PRIMARY KEY, order_id CHAR(26) NOT NULL, user_id VARCHAR(128) NOT NULL,
  change_type VARCHAR(32) NOT NULL, change_summary_json JSON NOT NULL,
  price_difference DECIMAL(12,2) NOT NULL DEFAULT 0, payment_method_id CHAR(26) NULL,
  action_id CHAR(26) NOT NULL UNIQUE, status VARCHAR(32) NOT NULL, created_at DATETIME(6) NOT NULL,
  INDEX idx_change_order (order_id,created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS rt_returns (
  return_id CHAR(26) PRIMARY KEY, order_id CHAR(26) NOT NULL, user_id VARCHAR(128) NOT NULL,
  reason VARCHAR(128) NOT NULL, status VARCHAR(32) NOT NULL, action_id CHAR(26) NOT NULL UNIQUE,
  version BIGINT UNSIGNED NOT NULL DEFAULT 1, created_at DATETIME(6) NOT NULL, updated_at DATETIME(6) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS rt_exchanges (
  exchange_id CHAR(26) PRIMARY KEY, order_id CHAR(26) NOT NULL, user_id VARCHAR(128) NOT NULL,
  reason VARCHAR(128) NOT NULL, status VARCHAR(32) NOT NULL, action_id CHAR(26) NOT NULL UNIQUE,
  version BIGINT UNSIGNED NOT NULL DEFAULT 1, created_at DATETIME(6) NOT NULL, updated_at DATETIME(6) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS rt_return_items (
  return_item_id CHAR(26) PRIMARY KEY, return_id CHAR(26) NOT NULL,
  order_item_id CHAR(26) NOT NULL, quantity INT UNSIGNED NOT NULL,
  created_at DATETIME(6) NOT NULL, UNIQUE KEY uk_return_order_item (return_id,order_item_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS rt_exchange_items (
  exchange_item_id CHAR(26) PRIMARY KEY, exchange_id CHAR(26) NOT NULL,
  order_item_id CHAR(26) NOT NULL, quantity INT UNSIGNED NOT NULL,
  created_at DATETIME(6) NOT NULL, UNIQUE KEY uk_exchange_order_item (exchange_id,order_item_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS rt_refunds (
  refund_id CHAR(26) PRIMARY KEY, order_id CHAR(26) NOT NULL, return_id CHAR(26) NULL,
  payment_id CHAR(26) NULL, user_id VARCHAR(128) NOT NULL, amount DECIMAL(12,2) NOT NULL,
  currency CHAR(3) NOT NULL, method VARCHAR(32) NOT NULL, status VARCHAR(32) NOT NULL,
  action_id CHAR(26) NOT NULL UNIQUE, external_refund_ref VARCHAR(255) NULL,
  version BIGINT UNSIGNED NOT NULL DEFAULT 1, created_at DATETIME(6) NOT NULL, updated_at DATETIME(6) NOT NULL,
  INDEX idx_refund_owner (user_id,status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS kb_documents (
  document_id CHAR(26) PRIMARY KEY, domain VARCHAR(32) NOT NULL, document_type VARCHAR(64) NOT NULL,
  title VARCHAR(512) NOT NULL, source_uri VARCHAR(2048) NOT NULL, version_no INT UNSIGNED NOT NULL,
  content_hash BINARY(32) NOT NULL, status VARCHAR(32) NOT NULL, effective_at DATETIME(6) NULL,
  expires_at DATETIME(6) NULL, metadata_json JSON NOT NULL, created_at DATETIME(6) NOT NULL,
  updated_at DATETIME(6) NOT NULL, UNIQUE KEY uk_kb_source_version (source_uri(255),version_no),
  INDEX idx_kb_domain_status (domain,document_type,status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS kb_chunks (
  chunk_id CHAR(26) PRIMARY KEY, document_id CHAR(26) NOT NULL, chunk_no INT UNSIGNED NOT NULL,
  section_path VARCHAR(1024) NULL, content MEDIUMTEXT NOT NULL, token_count INT UNSIGNED NOT NULL,
  content_hash BINARY(32) NOT NULL, index_status VARCHAR(32) NOT NULL,
  created_at DATETIME(6) NOT NULL, updated_at DATETIME(6) NOT NULL,
  UNIQUE KEY uk_kb_chunk (document_id,chunk_no), FULLTEXT KEY ft_kb_content (content)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS cs_governed_actions (
  action_id CHAR(26) PRIMARY KEY, user_id VARCHAR(128) NOT NULL, session_id VARCHAR(128) NOT NULL,
  turn_id VARCHAR(64) NOT NULL, tool_name VARCHAR(128) NOT NULL, tool_version VARCHAR(32) NOT NULL,
  skill_name VARCHAR(128) NULL, skill_version VARCHAR(32) NULL, arguments_json JSON NOT NULL,
  arguments_digest CHAR(64) NOT NULL, impact_summary VARCHAR(1000) NOT NULL, status VARCHAR(32) NOT NULL,
  idempotency_key VARCHAR(128) NOT NULL UNIQUE, resource_version BIGINT NULL,
  receipt_json JSON NULL, error_code VARCHAR(128) NULL, created_at DATETIME(6) NOT NULL,
  updated_at DATETIME(6) NOT NULL, expires_at DATETIME(6) NOT NULL, executed_at DATETIME(6) NULL,
  INDEX idx_action_owner_active (user_id,session_id,status,expires_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS cs_tool_call_receipts (
  receipt_id CHAR(26) PRIMARY KEY, action_id CHAR(26) NOT NULL, user_id VARCHAR(128) NOT NULL,
  tool_name VARCHAR(128) NOT NULL, idempotency_key VARCHAR(128) NOT NULL UNIQUE,
  status VARCHAR(32) NOT NULL, receipt_json JSON NOT NULL, created_at DATETIME(6) NOT NULL,
  INDEX idx_receipt_action (action_id), INDEX idx_receipt_owner (user_id,created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS cs_action_outbox (
  event_id CHAR(26) PRIMARY KEY, action_id CHAR(26) NOT NULL, event_type VARCHAR(64) NOT NULL,
  payload_json JSON NOT NULL, status VARCHAR(32) NOT NULL DEFAULT 'pending', attempts INT UNSIGNED NOT NULL DEFAULT 0,
  available_at DATETIME(6) NOT NULL, lease_until DATETIME(6) NULL, created_at DATETIME(6) NOT NULL,
  UNIQUE KEY uk_action_event (action_id,event_type), INDEX idx_action_outbox_claim (status,available_at,lease_until)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS cs_durable_checkpoints (
  thread_id VARCHAR(300) NOT NULL, checkpoint_ns VARCHAR(128) NOT NULL DEFAULT '',
  checkpoint_id VARCHAR(128) NOT NULL, parent_checkpoint_id VARCHAR(128) NULL,
  checkpoint_blob LONGBLOB NOT NULL, metadata_blob LONGBLOB NOT NULL,
  created_at DATETIME(6) NOT NULL, PRIMARY KEY(thread_id,checkpoint_ns,checkpoint_id),
  INDEX idx_checkpoint_latest (thread_id,checkpoint_ns,created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS cs_task_runs (
  task_run_id CHAR(26) PRIMARY KEY, turn_id VARCHAR(64) NOT NULL, task_id VARCHAR(64) NOT NULL,
  parent_task_id VARCHAR(64) NULL, domain VARCHAR(32) NOT NULL, capability VARCHAR(128) NOT NULL,
  status VARCHAR(32) NOT NULL, input_digest CHAR(64) NOT NULL, result_json JSON NULL,
  policy_version VARCHAR(32) NOT NULL, skill_version VARCHAR(32) NULL, tool_version VARCHAR(32) NULL,
  started_at DATETIME(6) NOT NULL, completed_at DATETIME(6) NULL,
  UNIQUE KEY uk_turn_task (turn_id,task_id), INDEX idx_task_trajectory (turn_id,started_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS cs_checkpoint_writes (
  thread_id VARCHAR(300) NOT NULL, checkpoint_ns VARCHAR(128) NOT NULL DEFAULT '',
  checkpoint_id VARCHAR(128) NOT NULL, task_id VARCHAR(128) NOT NULL,
  write_index INT UNSIGNED NOT NULL, channel VARCHAR(255) NOT NULL, value_blob LONGBLOB NOT NULL,
  PRIMARY KEY(thread_id,checkpoint_ns,checkpoint_id,task_id,write_index)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
