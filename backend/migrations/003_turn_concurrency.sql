CREATE TABLE IF NOT EXISTS cs_turn_leases (
    user_id VARCHAR(128) NOT NULL,
    session_id VARCHAR(128) NOT NULL,
    turn_id VARCHAR(64) NOT NULL,
    owner_id VARCHAR(64) NOT NULL,
    fencing_token BIGINT UNSIGNED NOT NULL,
    lease_until DATETIME(6) NOT NULL,
    stop_requested TINYINT(1) NOT NULL DEFAULT 0,
    updated_at DATETIME(6) NOT NULL,
    PRIMARY KEY (user_id, session_id),
    UNIQUE KEY uk_cs_turn_leases_turn (turn_id),
    INDEX idx_cs_turn_leases_expiry (lease_until)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
