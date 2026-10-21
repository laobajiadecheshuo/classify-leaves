import os, sys, threading, time
import requests

TOKEN = os.environ.get("KAGGLE_API_TOKEN")
if not TOKEN:
    sys.exit("[dl] 缺少环境变量 KAGGLE_API_TOKEN（kagglehub 的 KGAT_ token）")
API = "https://www.kaggle.com/api/v1/competitions/data/download-all/classify-leaves"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classify-leaves.zip")
NUM_CONN = 16
CHUNK = 4 * 1024 * 1024  # 4MB per read


def get_signed_url():
    r = requests.get(API, headers={"Authorization": f"Bearer {TOKEN}"},
                     allow_redirects=False, timeout=60)
    loc = r.headers.get("Location")
    if not loc:
        raise RuntimeError(f"no redirect, status {r.status_code}: {r.text[:200]}")
    return loc


def get_size(url):
    h = requests.head(url, allow_redirects=True, timeout=60)
    return int(h.headers["Content-Length"])


def main():
    url = get_signed_url()
    size = get_size(url)
    print(f"[dl] size = {size} bytes ({size/1024/1024:.1f} MB), conns = {NUM_CONN}", flush=True)

    with open(OUT, "wb") as f:
        f.truncate(size)

    done = [0] * NUM_CONN
    lock = threading.Lock()
    errors = []
    t0 = time.time()

    def worker(idx):
        start = idx * (size // NUM_CONN)
        end = size - 1 if idx == NUM_CONN - 1 else (idx + 1) * (size // NUM_CONN) - 1
        if start > end:
            return
        try:
            with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                              stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(OUT, "r+b") as f:
                    pos = start
                    for data in r.iter_content(CHUNK):
                        if not data:
                            continue
                        f.seek(pos)
                        f.write(data)
                        pos += len(data)
                        with lock:
                            done[idx] += len(data)
        except Exception as e:
            with lock:
                errors.append((idx, repr(e)))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(NUM_CONN)]
    for t in threads:
        t.start()

    while any(t.is_alive() for t in threads):
        time.sleep(2)
        got = sum(done)
        el = time.time() - t0
        spd = got / el / 1024 if el > 0 else 0
        pct = got / size * 100
        print(f"[dl] {pct:5.1f}%  {got/1024/1024:.1f}/{size/1024/1024:.1f} MB  "
              f"{spd:.0f} KB/s", flush=True)

    for t in threads:
        t.join()

    got = sum(done)
    if errors:
        print("[dl] ERRORS:", errors)
    print(f"[dl] done. got={got} size={size} match={got == size}")
    if got == size:
        print("[dl] SUCCESS:", OUT)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
