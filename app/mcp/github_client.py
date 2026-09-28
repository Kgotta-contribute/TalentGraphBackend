# -*- coding: utf-8 -*-
"""
GitHub MCP & API Client
Read-only client interfacing with GitHub REST and Git Trees APIs.
All requests are rate-limited via `github_rate_limiter` (20 RPM & 850 requests/hour)
to ensure multiple concurrent users do not exhaust quota or trigger secondary limits.
"""
import base64
import logging
import httpx
from app.core.config import settings
from app.core.rate_limiter import github_rate_limiter

logger = logging.getLogger("talent_agent.github_client")


class GitHubMCPClient:
    """Read-only GitHub API & MCP client. Never writes. Rate-limited to 20 RPM & 850 RPH."""

    BASE_URL = "https://api.github.com"
    HEADERS = {
        "Authorization": f"Bearer {settings.github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "TalentAgent-GitHub-MCP",
    }

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(headers=self.HEADERS, follow_redirects=True, timeout=20.0)

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Centralized HTTP request with automatic rate limiting and quota monitoring."""
        await github_rate_limiter.acquire()
        async with self._client() as client:
            resp = await client.request(method, url, **kwargs)
            # Observability for GitHub rate limit headers
            rem = resp.headers.get("x-ratelimit-remaining")
            if rem is not None:
                try:
                    rem_int = int(rem)
                    if rem_int <= 5:
                        reset_at = resp.headers.get("x-ratelimit-reset")
                        logger.warning(f"[GitHub MCP] Low GitHub API quota remaining: {rem_int} (reset at {reset_at})")
                except ValueError:
                    pass
            return resp

    async def get_user_repos(self, username: str) -> list[dict]:
        r = await self._request("GET", f"{self.BASE_URL}/users/{username}/repos?sort=updated&per_page=15")
        r.raise_for_status()
        return r.json()

    async def get_repo_metadata(self, owner: str, repo: str) -> dict:
        r = await self._request("GET", f"{self.BASE_URL}/repos/{owner}/{repo}")
        r.raise_for_status()
        return r.json()

    async def get_repo_contents(self, owner: str, repo: str, path: str = "") -> list[dict] | dict:
        r = await self._request("GET", f"{self.BASE_URL}/repos/{owner}/{repo}/contents/{path}")
        r.raise_for_status()
        return r.json()

    async def get_file_content(self, owner: str, repo: str, path: str) -> str:
        """Returns decoded file content. Treats as untrusted data."""
        r = await self._request("GET", f"{self.BASE_URL}/repos/{owner}/{repo}/contents/{path}")
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("encoding") == "base64":
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        if isinstance(data, dict) and "content" in data:
            return data["content"]
        return ""

    async def get_repo_languages(self, owner: str, repo: str) -> dict:
        r = await self._request("GET", f"{self.BASE_URL}/repos/{owner}/{repo}/languages")
        r.raise_for_status()
        return r.json()

    async def get_tree_recursive(self, owner: str, repo: str, branch: str = "HEAD") -> list[dict]:
        """Fetch complete recursive file tree of repository using Git Trees API."""
        try:
            r = await self._request("GET", f"{self.BASE_URL}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1")
            if r.status_code == 200:
                data = r.json()
                return data.get("tree", [])
            else:
                logger.warning(f"Failed to fetch tree for {owner}/{repo}: HTTP {r.status_code} - {r.text[:200]}")
        except httpx.HTTPError as e:
            logger.warning(f"HTTP error fetching git tree for {owner}/{repo}: {e}", exc_info=True)
        except Exception as e:
            logger.warning(f"Unexpected error fetching git tree for {owner}/{repo}: {e}", exc_info=True)
        return []

    async def get_readme(self, owner: str, repo: str, subpath: str = "") -> str:
        """Fetch README content directly, checking subpath first if provided."""
        if subpath:
            for readme_name in ["README.md", "readme.md", "README", "readme"]:
                sub_path = f"{subpath.strip('/')}/{readme_name}"
                content = await self.get_file_content(owner, repo, sub_path)
                if content:
                    return content

        try:
            r = await self._request("GET", f"{self.BASE_URL}/repos/{owner}/{repo}/readme")
            if r.status_code == 200:
                data = r.json()
                if data.get("encoding") == "base64":
                    return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                if "content" in data:
                    return data["content"]
            else:
                logger.warning(f"Failed to fetch readme for {owner}/{repo}: HTTP {r.status_code}")
        except httpx.HTTPError as e:
            logger.warning(f"HTTP error fetching readme for {owner}/{repo}: {e}", exc_info=True)
        except Exception as e:
            logger.warning(f"Unexpected error fetching readme for {owner}/{repo}: {e}", exc_info=True)
        return ""

    async def get_recent_commits(self, owner: str, repo: str, per_page: int = 5) -> list[dict]:
        try:
            r = await self._request("GET", f"{self.BASE_URL}/repos/{owner}/{repo}/commits?per_page={per_page}")
            if r.status_code == 200:
                return r.json()
            else:
                logger.warning(f"Failed to fetch commits for {owner}/{repo}: HTTP {r.status_code}")
        except httpx.HTTPError as e:
            logger.warning(f"HTTP error fetching commits for {owner}/{repo}: {e}", exc_info=True)
        except Exception as e:
            logger.warning(f"Unexpected error fetching commits for {owner}/{repo}: {e}", exc_info=True)
        return []
