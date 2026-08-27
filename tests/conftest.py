from __future__ import annotations

import os


# Encryption is mandatory outside debug mode. Tests use an explicit stable key
# so they do not depend on a developer's ignored .env file.
os.environ.setdefault("PII_ENCRYPTION_KEY", "smart-cs-agent-test-only-encryption-key")

