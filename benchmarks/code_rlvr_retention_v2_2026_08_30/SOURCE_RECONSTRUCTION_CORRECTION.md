# Source reconstruction correction

Frozen before runtime smoke, formal model training, or any v2 model evaluation.

## Error found by the equivalence gate

The first converted-parquet reconstruction assumed that `0000.parquet` and
`0001.parquet` supplied the requested 1,600 rows. Their actual row counts are
500 and 1,000, so the iterator exhausted at `rows_seen=1500`. That incomplete
view produced different historical train/dev hashes and could not be accepted.
The temporary v2.1 protocol based on that view is retracted, not deleted.

## Correct official source prefix

The Dataset Server 1,600-row view is reconstructed from all 500 rows of
`0000.parquet`, all 1,000 rows of `0001.parquet`, and the first 100 rows of
`0002.parquet`, in that order. The pinned official converted files are:

| file | rows consumed | bytes | SHA-256 |
| --- | ---: | ---: | --- |
| `0000.parquet` | 500 | 398,180,786 | `9a6741885dc8556c51d377f2b2fb38bb8f587bc149e3f3a62939ee8d946290be` |
| `0001.parquet` | 1,000 | 497,224,109 | `2f94ddd239a3e0974219f963fb2407e4fdfd341e8c977edd92e1094abb6ea7b4` |
| `0002.parquet` | first 100 | 208,056,834 | `3590b6e2459160e03c4a1ac699de3d1e79f8ac4096c6cd2e824713a20ca79c47` |

## Accepted split receipt

The corrected reconstruction passed every equivalence and feasibility gate:

- rows seen: 1,600
- admitted candidates: 1,434
- historical train: 800, SHA-256
  `6cf24bc6da5bd2111c6b4ba730fa679b009ecd5808507cd7484fd903dbc2ec1e`
- historical dev: 200, SHA-256
  `67c9fb2f5bbba31ae3886f49dbcdc98d7231784c559103e328fa931b3e418bda`
- new development: 200, SHA-256
  `a59ed4871f2189d728054959971d245a9358cbc32e62722b5edaa9ca8e486a88`
- new final: 200, SHA-256
  `723b9c30abb3b9915c01e7f98eb70800f7d23ca201a7fd7b155019f2b8c50902`
- holdout near-duplicates removed: 0 at the unchanged threshold `<0.85`

The two historical hashes exactly match the 2026-08-29 campaign. Therefore the
original v2 protocol is feasible and authoritative; no split-size or threshold
change is made. The invalid 1,500-row files are not used by any model job.
