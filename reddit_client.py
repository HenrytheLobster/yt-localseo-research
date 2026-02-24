
"""reddit_client.py - Reddit OAuth + API wrapper (v1)

Official Reddit API via OAuth (no scraping). Minimal dependency: requests.

Auth:
- Uses client_credentials grant (read-only) by default.
- Requires: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

Notes:
- Reddit Data API rate limit guidance: ~100 requests/min per client id.
"""

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from utils import AgentError

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
API_BASE = "https://oauth.reddit.com"


@dataclass
class RedditAuth:
    client_id: str
    client_secret: str
    user_agent: str


class RedditClient:
    def __init__(self, auth: RedditAuth, timeout: int = 30):
        self.auth = auth
        self.timeout = timeout
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0

    @staticmethod
    def from_env() -> "RedditClient":
        cid = os.environ.get("REDDIT_CLIENT_ID", "").strip()
        secret = os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
        ua = os.environ.get("REDDIT_USER_AGENT", "").strip()
        if not (cid and secret and ua):
            raise AgentError(
                "Missing Reddit credentials. Set in environment/.env:\n"
                "  REDDIT_CLIENT_ID=...\n"
                "  REDDIT_CLIENT_SECRET=...\n"
                "  REDDIT_USER_AGENT=your-app/0.1 (contact: you@example.com)\n"
            )
        return RedditClient(RedditAuth(cid, secret, ua))

    def get_token(self) -> str:
        if self._token and time.time() < (self._token_expires_at - 60):
            return self._token

        data = {"grant_type": "client_credentials"}
        headers = {"User-Agent": self.auth.user_agent}
        try:
            resp = requests.post(
                TOKEN_URL,
                auth=(self.auth.client_id, self.auth.client_secret),
                data=data,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            self._token = payload.get("access_token")
            expires_in = float(payload.get("expires_in", 3600))
            if not self._token:
                raise AgentError(f"Token response missing access_token: {payload}")
            self._token_expires_at = time.time() + expires_in
            return self._token
        except requests.RequestException as e:
            raise AgentError(f"Reddit OAuth token fetch failed: {e}")

    def api_get(self, path: str, params: Optional[Dict[str, Any]] = None, retries: int = 3) -> Any:
        url = f"{API_BASE}{path}"
        backoff = [2, 6, 15]

        for attempt in range(retries + 1):
            token = self.get_token()
            headers = {
                "Authorization": f"bearer {token}",
                "User-Agent": self.auth.user_agent,
            }
            try:
                resp = requests.get(url, headers=headers, params=params or {}, timeout=self.timeout)

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait = int(retry_after) if retry_after and retry_after.isdigit() else backoff[min(attempt, len(backoff)-1)]
                    time.sleep(wait)
                    continue

                if resp.status_code == 401 and attempt < retries:
                    self._token = None
                    self._token_expires_at = 0.0
                    time.sleep(1)
                    continue

                if resp.status_code >= 500 and attempt < retries:
                    time.sleep(backoff[min(attempt, len(backoff)-1)])
                    continue

                resp.raise_for_status()
                return resp.json()

            except requests.RequestException as e:
                if attempt < retries:
                    time.sleep(backoff[min(attempt, len(backoff)-1)])
                    continue
                raise AgentError(f"Reddit GET failed after {retries+1} attempts: {e}")

        raise AgentError("Reddit GET failed (unexpected fallthrough)")
