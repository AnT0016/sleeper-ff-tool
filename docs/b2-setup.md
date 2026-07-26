# Backblaze B2 setup (the production lake backend)

The lake has two interchangeable storage backends behind one `StorageBackend` protocol
([src/store/lake.py](../src/store/lake.py)):

| `LAKE_BACKEND` | Backend | Where | Used by |
| --- | --- | --- | --- |
| `local` (default) | `LocalParquetBackend` | `data_cache/lake/` (gitignored) | local runs, tests, anyone without creds |
| `s3` | `S3Backend` ([src/store/s3.py](../src/store/s3.py)) | a private B2 bucket | the collection crons (#6) |

Nothing is vendor-specific. B2 is just the endpoint we point at — moving to R2/AWS/Tigris later is a
different `LAKE_S3_ENDPOINT`, not a rewrite. B2 was chosen because its 10 GB free tier needs **no
credit card** (Cloudflare R2 requires completing a payment method even on the free tier) and because
it speaks the S3 API. See the decision log in [docs/plans/data-collection.md](plans/data-collection.md).

## Owner checklist (one-time, manual — account creation cannot be automated)

> ✅ Already done. Recorded here so it is reproducible if the account is ever rebuilt.

1. **Create the Backblaze account, choosing EU Central (Amsterdam) at signup.** The region is fixed
   **per account, permanently** — it cannot be changed afterwards. EU Central because the heavy
   transfers (the 2016–2025 backfill, and repeated `build_training_frame` reads during model work)
   run from the owner's machine in France; the cron writes that would favour a US region are small
   and unattended.
2. **Create a private bucket.** Files: **Private**. Default encryption: **enabled**. Object Lock:
   **disabled** — it would block the read-modify-write that `write_snapshot` performs on a partition.
3. **Create a bucket-scoped S3 application key** with Read+Write on that bucket only. Backblaze shows
   the secret **once**; copy it immediately.
4. **Record the four values** as GitHub Actions repository secrets, and in your local user
   environment if you want to read the bucket from your machine:

   | Secret | Example | Notes |
   | --- | --- | --- |
   | `LAKE_S3_ENDPOINT` | `https://s3.eu-central-003.backblazeb2.com` | full URL, shown on the bucket page |
   | `LAKE_S3_ACCESS_KEY_ID` | the application **keyID** | not the master key |
   | `LAKE_S3_SECRET_ACCESS_KEY` | the application key | shown once |
   | `LAKE_S3_BUCKET` | the bucket name | |

   `LAKE_S3_REGION` is optional: the region is read off the endpoint host
   (`s3.**eu-central-003**.backblazeb2.com`), and only needs setting for a vendor that doesn't
   encode it there.

No credit card, and no secret ever enters the repo — the immutable "no secrets in the repo" rule
holds, and the playbook explicitly permits Actions secrets.

## Using it

```bash
LAKE_BACKEND=s3 python scripts/collect.py --mode postgame
```

With `LAKE_BACKEND` unset everything keeps using `data_cache/lake/`, so no local workflow and no test
depends on the bucket existing.

**Populating the bucket from an existing local lake** — the one-time re-run after this backend lands:

```bash
LAKE_BACKEND=s3 python scripts/backfill_lake.py --seasons 2016-2025
```

Partition keys are identical on both backends (`<source>/season=<YYYY>/<file>.parquet`), so a bucket
populated this way is byte-for-byte the same layout as the local materialization.

## Behaviour worth knowing

- **Missing or blank credentials raise `S3ConfigError`** naming the absent variable. The backend
  never falls back to local: a cron that silently wrote to a container-local `data_cache/lake/` would
  report success every week and accumulate nothing, and nothing downstream would notice.
- **A mistyped `LAKE_S3_BUCKET` raises too.** S3 answers a `HEAD` against a missing bucket with a
  404, the same status as a missing object, so the backend distinguishes them by error code —
  otherwise a typo would make every partition read as "not captured yet".
- **Writes are a single `PutObject`** of the fully merged partition. Object stores have no rename, so
  this is the substitute for the local backend's temp-file-plus-`os.replace`: readers see either the
  old object or the new one, and an interrupted run leaves nothing to clean up.
- **Metadata reads are cheap.** `lake_inventory()` and column-projected reads fetch the parquet
  footer and the requested column chunks over HTTP range requests, never the whole object. This is
  load-bearing: the lake is 412 partitions today, and a backend that answered by downloading them
  would make routine calls cost the entire lake.
- **Costs.** B2 free tier: 10 GB storage, 1 GB/day egress, unlimited upload. A decade of this lake is
  well under 1 GB, and the crons only ever write.
