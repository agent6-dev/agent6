# Go cache reuse

Fresh `agent6 exec --new` calls reuse Go's build cache under `partial` isolation with either `sandbox.home = "auto"` or `"cache"`.
The reported cold cache on every run did not reproduce at that isolation level.
`full` could not be exercised because the enclosing sandbox blocks unprivileged user namespaces.

## Method

- Go 1.26.8 on Linux amd64, Landlock ABI 8.
- One dependency-free Go module with a test that checks `2 + 2 == 4`.
- `GOMAXPROCS=2 go test -count=1 ./...`, timed inside the command with `/usr/bin/time`, excluding agent6 startup.
- Three cold/warm pairs per mode, each pair with a fresh cache root and two successive invocations.
- The direct control ran outside the nested `agent6 exec`, but inside the enclosing agent sandbox.
- Modes ran sequentially, direct control, `auto`, then `cache`.
- This was a cache-reuse diagnostic, not a controlled performance benchmark; host contention was not controlled, and scoped Python tests overlapped the final cache pair.

`-count=1` disables test-result reuse, not compiled-package reuse.
Inside the jail, `go env GOCACHE` resolved to `.cache/go-build` beneath the repository's persistent jail HOME.

## Results

All 18 commands exited 0.
Times are seconds, in replicate order.

| Mode | Cold | Warm | Cold median | Warm median |
|---|---|---|---|---|
| Direct control | 36.58, 32.41, 38.31 | 0.64, 0.56, 0.66 | 36.58 | 0.64 |
| `partial`, `home = "auto"` | 32.92, 37.10, 28.60 | 2.54, 0.86, 0.68 | 32.92 | 0.86 |
| `partial`, `home = "cache"` | 32.07, 36.50, 31.69 | 0.57, 0.62, 0.70 | 32.07 | 0.62 |

## Default

The default stays `auto`: temporary under `full`, persistent where private tmpfs is unavailable.
An existing cache does not establish that the operator accepts cross-run persistence of executable files and tool configuration.
The explicit `cache` setting already provides reuse without granting the operator's own Go cache.
See [Go build cache](../docs/usage.md#go-build-cache) for the setup.
