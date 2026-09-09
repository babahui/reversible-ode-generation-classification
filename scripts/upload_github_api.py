#!/usr/bin/env python3
"""Publish the tracked HEAD tree through GitHub's Git Data REST API."""

import argparse
import base64
import getpass
import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def git(*arguments: str, cwd: Path, binary: bool = False):
    result = subprocess.run(
        ("git", *arguments), cwd=cwd, check=True, capture_output=True
    ).stdout
    return result if binary else result.decode("utf-8").strip()


class GitHubAPI:
    def __init__(self, token: str, repository: str):
        self.token = token
        self.base = f"https://api.github.com/repos/{repository}"

    def request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        allowed_errors: Tuple[int, ...] = (),
    ) -> Tuple[int, Dict[str, Any]]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            self.base + path,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
                "User-Agent": "reversible-ode-release-uploader",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
                return response.status, json.loads(data) if data else {}
        except urllib.error.HTTPError as error:
            data = error.read()
            parsed = json.loads(data) if data else {}
            if error.code in allowed_errors:
                return error.code, parsed
            message = parsed.get("message", str(error))
            raise RuntimeError(f"GitHub API {method} {path}: HTTP {error.code}: {message}") from error


def tracked_tree(root: Path):
    raw = git("ls-tree", "-r", "-z", "HEAD", cwd=root, binary=True)
    entries = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, path_bytes = record.split(b"\t", 1)
        mode, object_type, _ = metadata.decode("ascii").split()
        if object_type != "blob":
            raise RuntimeError(f"Unsupported Git object type: {object_type}")
        path = path_bytes.decode("utf-8")
        content = git("show", f"HEAD:{path}", cwd=root, binary=True).decode("utf-8")
        entries.append({"path": path, "mode": mode, "type": "blob", "content": content})
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, help="GitHub owner/name")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Upload committed HEAD while leaving local uncommitted changes untouched.",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    dirty = git("status", "--porcelain", cwd=root)
    if dirty and not args.allow_dirty:
        raise RuntimeError("Refusing to upload a dirty working tree; commit changes first")
    if dirty:
        print("Local uncommitted changes are present; uploading committed HEAD only.", flush=True)
    token = os.environ.get("GITHUB_TOKEN", "").strip() or getpass.getpass("GitHub token: ")
    if not token:
        raise RuntimeError("GitHub token is required")
    api = GitHubAPI(token, args.repository)
    status, repository = api.request("GET", "")
    if status != 200 or not repository.get("permissions", {}).get("push"):
        raise RuntimeError("Token does not have push permission for this repository")

    ref_path = f"/git/ref/heads/{args.branch}"
    ref_status, ref = api.request("GET", ref_path, allowed_errors=(404, 409))
    if ref_status == 409:
        # GitHub refuses Git Data tree creation until an empty repository has
        # its first object. The next full-tree commit removes this marker.
        api.request(
            "PUT",
            "/contents/.github-api-bootstrap",
            {
                "message": "Initialize repository for API upload",
                "content": base64.b64encode(b"temporary\n").decode("ascii"),
                "branch": args.branch,
            },
        )
        ref_status, ref = api.request("GET", ref_path)

    entries = tracked_tree(root)
    print(f"Uploading {len(entries)} tracked files through GitHub API...", flush=True)
    _, tree = api.request("POST", "/git/trees", {"tree": entries})
    parents = [ref["object"]["sha"]] if ref_status == 200 else []
    _, commit = api.request(
        "POST",
        "/git/commits",
        {
            "message": git("log", "-1", "--format=%B", cwd=root),
            "tree": tree["sha"],
            "parents": parents,
        },
    )
    if ref_status == 200:
        api.request("PATCH", f"/git/refs/heads/{args.branch}", {"sha": commit["sha"]})
    else:
        api.request(
            "POST", "/git/refs", {"ref": f"refs/heads/{args.branch}", "sha": commit["sha"]}
        )
    print(
        json.dumps(
            {
                "repository": repository["html_url"],
                "branch": args.branch,
                "commit": commit["sha"],
                "tracked_files": len(entries),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
