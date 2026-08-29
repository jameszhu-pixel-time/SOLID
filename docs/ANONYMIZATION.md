# Anonymization and release audit

## Removed

- personal usernames and research-development initials;
- personal or cluster-specific absolute paths;
- dated assistant/developer marker comments;
- experiment-account, team, and project defaults;
- external experiment-tracking account and run links;
- experiment tracking as a default logger in shipped configurations;
- inherited experiment-tracking credentials in launch environments;
- hard-coded authorization examples;
- checkpoints, logs, generated outputs, datasets, models, solver licenses, and
  runtime caches;
- non-English research-specific comments and report labels.

## Preserved

Upstream copyright headers, the Apache-2.0 license, and third-party attribution
notices are preserved because they are legal provenance, not research-author
metadata. The `verl` import namespace is also preserved for compatibility.

Generic framework interfaces that accept an API key remain interfaces only:
the repository contains no populated key, token, account, or endpoint secret.
External experiment tracking remains available in upstream framework modules,
but the SOLID launchers disable it and use console-only logging.

## Release checks

Before release, run:

```bash
bash scripts/audit_release.sh
```

The audit rejects:

- cluster-specific absolute paths;
- CJK annotations;
- full-width annotation punctuation;
- external tracking-account links and tracking-enabled logger defaults;
- common literal secret assignments;
- tracked runtime artifacts and credential-like files;
- Python syntax errors in the package and release scripts.
