# Documents Service Runbook

The documents service accepts uploads, stores source files, and asynchronously indexes searchable text.

## Queue thresholds

A pending indexing queue above 5,000 documents is a warning condition. A queue above 20,000 documents is critical and requires the Documents Platform on-call engineer.

## Reindexing

Before a full reindex, pause normal indexing consumers, record the current queue depth, and verify source objects are available. Resume normal consumers only after the reindex worker is stable and the backlog is decreasing.

## User impact

Uploads can remain available even when indexing is delayed. During an indexing incident, tell users that newly uploaded documents may not appear in search until the backlog has been processed.
