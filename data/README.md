# PSAE-SSL benchmark datasets

Each folder contains:
- `edges.txt` — one edge per line: `src dst` (1-indexed node ids)
- `labels.txt` — one integer community label per line (line i = label of node i)

| Dataset   | Nodes  | Edges    | Communities | Notes |
|-----------|--------|----------|-------------|-------|
| karate    | 34     | 78       | 2           | Zachary 1977 |
| dolphins  | 62     | 159      | 2           | Lusseau et al. |
| football  | 115    | 613      | 12          | Girvan–Newman conferences |
| polbooks  | 105    | 441      | 3           | Krebs (l/n/c) |
| polblogs  | 1490   | 19022    | 2           | Adamic–Glance (directed) |
| email     | 1005   | 24929    | 42          | SNAP email-Eu-core (directed) |
| cora      | 2708   | 5278     | 7           | Citation network |
| citeseer  | 3264   | 4536     | 6           | Citation network (directed) |
| pubmed    | 19717  | 44324    | 3           | Citation network |
| dblp      | 317080 | 1049866  | 13477*      | SNAP com-DBLP; *K forced to 13477 in code |

All datasets are pre-installed. `usr.autoDownload = false` in get_user_settings.m.
