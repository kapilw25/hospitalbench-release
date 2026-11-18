#!/usr/bin/env python3
"""fetch_assets.py: download the NVIDIA SimReady USD assets the chores use, from their pinned public URLs, into a local cache. No token, no account.

    python scripts/fetch_assets.py                  # every asset in assets_manifest.json (about a few GB; --list shows the size first)
    python scripts/fetch_assets.py --task T3        # only the assets chore T3 uses (repeat --task for several)
    python scripts/fetch_assets.py --list           # names, file counts and bytes; downloads nothing
    python scripts/fetch_assets.py --verify         # re-hash every cached file against the pinned revision

Source: https://huggingface.co/datasets/nvidia/simready-assets at the commit pinned in assets_manifest.json (one revision for all assets, so a file can never change
under you). Each asset is a package folder <asset>/ (its simready_usd/ entry point, physics layers, materials, textures); the whole folder is fetched, the same set
the Isaac Lab-Arena loader fetches.

Cache: $HB_ASSET_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub, in the Hugging Face hub layout
    <cache>/datasets--nvidia--simready-assets/snapshots/<revision>/<asset>/simready_usd/...
so Isaac Lab-Arena (hf_hub_download) and the scene scripts find the files without a second download.

Checks, per file: the size and the hash the hub lists for the pinned revision (SHA-256 for large files, the git blob SHA-1 for the others) must match before the file
is moved into the cache. A file already in the cache is accepted when its size matches; --verify re-hashes it. Transient errors (timeouts, HTTP 429/5xx) are retried
with growing waits (--retries, default 4). Offline or blocked network: the script says so, names the cache folder and exits 2; nothing is left half written.
Exit codes: 0 all present and verified, 1 a file failed its check, 2 network unreachable, 3 bad arguments or manifest."""
import argparse, hashlib, json, os, re, socket, sys, time, urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://huggingface.co/api/datasets/{repo}/tree/{rev}/{path}?recursive=true"
RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
UA = {"User-Agent": "hospitalbench-fetch-assets"}


def cache_root():
    c = os.environ.get("HB_ASSET_CACHE") or os.path.join(os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface"), "hub")
    return os.path.abspath(c)


class Offline(Exception): pass
class Bad(Exception): pass


def http(url, retries, timeout=60):
    """GET -> (bytes or open response). Retries 429/5xx/timeouts; connection failures on the first try raise Offline."""
    last = None
    for i in range(retries + 1):
        try: return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504): last = e
            else: raise Bad(f"HTTP {e.code} for {url}")
        except (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError) as e:
            last = e
            if isinstance(e, urllib.error.URLError) and isinstance(e.reason, socket.gaierror): raise Offline(str(e.reason))
        time.sleep(min(60, 2 ** (i + 1)))
    if isinstance(last, (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError)): raise Offline(str(last))
    raise Bad(f"gave up after {retries + 1} tries: {last}")


def list_package(repo, rev, folder, retries):
    """files of one package folder at the pinned revision: [{path,size,sha256|None,oid}] (the hub tree API, paged by Link headers)"""
    url, out = API.format(repo=repo, rev=rev, path=urllib.parse.quote(folder)), []
    while url:
        r = http(url, retries)
        out += json.loads(r.read().decode())
        m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", "") or "")
        url = m.group(1) if m else None
    files = []
    for e in out:
        if e.get("type") != "file": continue
        p = e["path"]
        if not p.startswith(folder + "/") or ".." in p.split("/"): raise Bad(f"unexpected path {p!r} under {folder}")
        files.append({"path": p, "size": int(e["size"]), "oid": e["oid"], "sha256": (e.get("lfs") or {}).get("oid")})
    if not files: raise Bad(f"{folder}: no files listed at revision {rev}")
    return files


def digest(path, f):
    """sha256 for large (LFS) files, else the git blob sha1 ('blob <n>\\0' + bytes), as the hub lists them"""
    h = hashlib.sha256() if f["sha256"] else hashlib.sha1()
    if not f["sha256"]: h.update(b"blob %d\0" % f["size"])
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def want(f): return f["sha256"] or f["oid"]


def fetch_file(repo, rev, f, dest, retries, verify):
    """-> 'cached' | 'downloaded'. Writes dest.part, checks size and hash, then renames."""
    if os.path.isfile(dest) and os.path.getsize(dest) == f["size"] and (not verify or digest(dest, f) == want(f)): return "cached"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    for attempt in range(retries + 1):
        try:
            r = http(RESOLVE.format(repo=repo, rev=rev, path=urllib.parse.quote(f["path"])), retries)
            with open(part, "wb") as fh:
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    fh.write(b)
            if os.path.getsize(part) != f["size"] or digest(part, f) != want(f):
                raise Bad(f"{f['path']}: size or hash differs from the pinned revision")
            os.replace(part, dest); return "downloaded"
        except (Bad, OSError, urllib.error.URLError, socket.timeout, ConnectionError) as e:
            if os.path.exists(part): os.remove(part)
            if isinstance(e, Bad) and "HTTP 4" in str(e): raise
            if attempt == retries: raise Bad(str(e))
            time.sleep(min(60, 2 ** (attempt + 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", action="append", help="only the assets of this chore (T1..T10), repeatable")
    ap.add_argument("--asset", action="append", help="only this asset name, repeatable")
    ap.add_argument("--list", action="store_true", help="list assets with file counts and bytes, download nothing")
    ap.add_argument("--verify", action="store_true", help="re-hash cached files")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", default=None, help="cache folder (default: see above)")
    ap.add_argument("--manifest", default=os.path.join(ROOT, "assets_manifest.json"))
    ap.add_argument("--quiet", action="store_true", help="print only problems and the final line")
    A = ap.parse_args()
    if A.cache: os.environ["HB_ASSET_CACHE"] = A.cache
    try: man = json.load(open(A.manifest))
    except (OSError, ValueError) as e: print(f"cannot read {A.manifest}: {e}"); return 3
    repo, rev = man["repo"], man["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", rev): print("assets_manifest.json: revision is not a 40-hex commit"); return 3
    names = sorted(man["assets"])
    if A.task: names = [n for n in names if set(t.upper() for t in A.task) & set(man["assets"][n]["chores"])]
    if A.asset: names = [n for n in names if n in A.asset]
    if not names: print("no asset matches --task/--asset"); return 3
    snap = os.path.join(cache_root(), "datasets--" + repo.replace("/", "--"), "snapshots", rev)
    say = (lambda *a: None) if A.quiet else print
    try:
        tree = {}
        with ThreadPoolExecutor(A.workers) as ex:
            futs = {ex.submit(list_package, repo, rev, n, A.retries): n for n in names}
            for fu in as_completed(futs): tree[futs[fu]] = fu.result()
        if A.list:
            tot = 0
            for n in names:
                b = sum(f["size"] for f in tree[n]); tot += b
                print(f"{n:48s} {len(tree[n]):4d} files {b / 1e6:9.1f} MB  chores {','.join(man['assets'][n]['chores'])}")
            print(f"{len(names)} assets, {tot / 1e9:.2f} GB at revision {rev[:12]}; cache {snap}"); return 0
        say(f"{len(names)} assets, revision {rev[:12]}, cache {snap}")
        jobs = [(n, f) for n in names for f in tree[n]]
        done = dl = 0; bad = []
        with ThreadPoolExecutor(A.workers) as ex:
            futs = {ex.submit(fetch_file, repo, rev, f, os.path.join(snap, *f["path"].split("/")), A.retries, A.verify): (n, f) for n, f in jobs}
            for fu in as_completed(futs):
                n, f = futs[fu]
                try: dl += fu.result() == "downloaded"
                except Bad as e: bad.append(f"{f['path']}: {e}")
                done += 1
                if not A.quiet and done % 200 == 0: print(f"  {done}/{len(jobs)} files", flush=True)
        for n in names:                      # the entry point named in the manifest must be there
            ent = os.path.join(snap, *man["assets"][n]["entry"].split("/"))
            if not os.path.isfile(ent): bad.append(f"{n}: entry point {man['assets'][n]['entry']} missing")
        if bad:
            print(f"FAILED: {len(bad)} problems"); [print("  " + b) for b in bad[:20]]; return 1
        print(f"assets OK: {len(names)} assets, {len(jobs)} files present ({dl} downloaded and hash-checked, {len(jobs) - dl} already cached" + (", re-hashed" if A.verify else ", size-checked") + f") in {snap}"); return 0
    except Offline as e:
        print(f"cannot reach huggingface.co ({e}). Offline, or the network blocks it. Nothing was changed. Cache folder: {snap}\n"
              f"Re-run this command when the network is up; files already cached are kept and skipped.")
        return 2
    except Bad as e:
        print(f"FAILED: {e}"); return 1


if __name__ == "__main__":
    sys.exit(main())
