"""Pull only the results member out of each HAL trace zip on Hugging Face, with HTTP range requests.

The 380 trace zips total about 113 GB because they carry full agent logs; the per-task results and costs sit in one
small member per zip, *_UPLOAD.json.encrypted. A zip's central directory is at the end of the file, so a seekable
HTTP reader lets Python's zipfile read that member alone (a few hundred KB) instead of the whole archive.

usage: python hal_fetch.py [--limit N] [--match SUBSTRING]
Writes data/hal/uploads/<zip-stem>.json (decrypted) and data/hal/index.json (what was fetched, sizes, errors).
"""
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hal_decrypt import decrypt_member  # noqa: E402
from hal_extract import extract  # noqa: E402

REPO = "https://huggingface.co/datasets/agent-evals/hal_traces"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "hal" / "runs"            # compact per-task records only; the raw uploads are not kept
MAX_MEMBER = 1_500_000_000                       # bytes; larger results members are recorded as skipped


class MemberTooLarge(Exception):
    pass


class HttpFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests (follows Hugging Face's redirects)."""

    def __init__(self, url, size=None, chunk=1 << 20):
        self.url = url
        self.pos = 0
        self.chunk = chunk
        self.size = size if size is not None else self._head()
        self.cache = {}

    def _head(self):
        req = urllib.request.Request(self.url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def _range(self, start, end):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        data = self._range(self.pos, end)
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def list_zips():
    with urllib.request.urlopen(f"{REPO.replace('huggingface.co/datasets', 'huggingface.co/api/datasets')}/tree/main", timeout=60) as r:
        tree = json.load(r)
    return [(x["path"], int(x.get("size", 0))) for x in tree if x["path"].endswith(".zip")]


def fetch_upload(name, size):
    url = f"{REPO}/resolve/main/{name}"
    f = HttpFile(url, size=size)
    with zipfile.ZipFile(f) as z:
        members = [i for i in z.infolist() if i.filename.endswith("UPLOAD.json.encrypted")] or \
                  [i for i in z.infolist() if i.filename.endswith(".encrypted")]
        if not members:
            raise ValueError(f"no encrypted member; first members: {[i.filename for i in z.infolist()[:3]]}")
        m = members[0]
        if m.file_size > MAX_MEMBER:                       # decrypting and parsing needs several times the member size in memory
            raise MemberTooLarge(f"results member is {m.file_size/1e9:.1f} GB; limit {MAX_MEMBER/1e9:.1f} GB")
        return decrypt_member(z.read(m)), m.file_size


def _read_index(path):
    for _ in range(5):
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except json.JSONDecodeError:              # another worker is mid-write
            time.sleep(0.5)
    return {}


def _write_entry(path, name, entry):
    """Merge one entry into the shared index (several workers may run at once)."""
    import os
    for attempt in range(6):                        # another worker may hold the file for a moment
        try:
            cur = _read_index(path)
            cur[name] = entry
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(cur, indent=1), encoding="utf-8")
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.5 * (attempt + 1))


def main(argv):
    limit = None
    match = None
    if "--limit" in argv:
        limit = int(argv[argv.index("--limit") + 1])
    if "--match" in argv:
        match = argv[argv.index("--match") + 1]
    OUT.mkdir(parents=True, exist_ok=True)
    index_path = ROOT / "data" / "hal" / "index.json"
    index = _read_index(index_path)
    zips = list_zips()
    if match:
        zips = [z for z in zips if match in z[0]]
    global MAX_MEMBER
    if "--max-member" in argv:                         # raise the member limit for a single-worker pass over the big ones
        MAX_MEMBER = int(float(argv[argv.index("--max-member") + 1]))
    retry_large = "--retry-large" in argv              # include entries skipped as MemberTooLarge under a lower limit
    refetch = "--refetch" in argv                      # redo entries made by an older extractor (index entry without "v": 2)

    def skipped(name):
        e = index.get(name, {})
        if not e.get("skip"):
            return False
        return not (retry_large and "MemberTooLarge" in (e.get("error") or ""))

    todo = [z for z in zips if (z[0] not in index or index[z[0]].get("error") or (refetch and index[z[0]].get("v") != 2))
            and not skipped(z[0])]
    if "--reverse" in argv:
        todo = todo[::-1]
    if limit:
        todo = todo[:limit]
    print(f"{len(zips)} zips match; {len(todo)} to fetch", flush=True)
    for k, (name, size) in enumerate(todo, 1):
        t0 = time.time()
        current = _read_index(index_path)
        if name in current and not current[name].get("error") and (not refetch or current[name].get("v") == 2):
            continue                                   # another worker got there first
        if current.get(name, {}).get("skip") and not retry_large:
            continue
        try:
            obj, msize = fetch_upload(name, size)
            stem = name[:-4]
            rec = extract(obj, name)
            del obj
            (OUT / f"{stem}.json").write_text(json.dumps(rec), encoding="utf-8")
            rep, rec_cost = rec["total_cost_reported"], rec["total_cost_recomputed"]
            entry = {"zip_bytes": size, "member_bytes": msize, "seconds": round(time.time() - t0, 1),
                     "benchmark": rec["benchmark"], "agent": rec["agent_name"], "tasks": len(rec["tasks"]),
                     "cost_reported": rep, "cost_recomputed": rec_cost, "unpriced": rec["unpriced_models"],
                     "v": 2, "duplicate_calls": rec["n_duplicate_calls"], "tokens_raw": rec["tokens_raw"], "tokens_counted": rec["tokens_counted"]}
            ratio = rec["tokens_raw"] / rec["tokens_counted"] if rec["tokens_counted"] else float("nan")
            print(f"[{k}/{len(todo)}] {name}  zip {size/1e6:.0f} MB  {time.time()-t0:.0f}s  tasks {len(rec['tasks'])}  "
                  f"cost {rep} vs {rec_cost:.2f}  dup {rec['n_duplicate_calls']} (raw/counted {ratio:.3f})"
                  + (f"  UNPRICED {rec['unpriced_models']}" if rec["unpriced_models"] else ""), flush=True)
        except BaseException as e:                     # pyo3's PanicException (out of memory in the decryptor) is not an Exception
            if isinstance(e, KeyboardInterrupt):
                raise
            entry = {"zip_bytes": size, "error": f"{type(e).__name__}: {e}"}
            if isinstance(e, MemberTooLarge):
                entry["skip"] = True                   # do not retry on the next run; the machine cannot hold it
            # a decryptor panic or MemoryError is left as a plain error, so a later single-worker pass retries it
            print(f"[{k}/{len(todo)}] {name}  FAILED {type(e).__name__}: {e}", flush=True)
        _write_entry(index_path, name, entry)


if __name__ == "__main__":
    main(sys.argv[1:])
