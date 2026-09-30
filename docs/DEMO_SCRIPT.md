# Three-minute demo script

Setup (before the room fills): `make setup`, then
`python -m loci.edge.ui --no-autosync --nodes 2` and open http://127.0.0.1:8765.
`--no-autosync` means the simulator robots sync only when you click, so the story is exactly what
happens; the two process devices (robot-c, robot-d) run on their own. If you have a free-tier model
download available, set `LOCI_EMBEDDER=fastembed` first for real semantic search.
Fallback if anything misbehaves: play [assets/edge-demo.webm](assets/edge-demo.webm) (about 2 min)
and talk over it.

Say up front: **"The robots' sensors are simulated. The Qdrant Edge shards, the device processes,
the sync, the decisions and every number on screen are real."**

## Beats

| # | Time | Do | Say |
|---|---|---|---|
| 1 | 0:00 | Point at the fleet column, click **robot-c**. | "Four devices. Two are scripted simulators; robot-c and robot-d are separate OS processes, each with its own Qdrant Edge shards on its own disk. This is their live PID, memory, CPU and Qdrant latency." |
| 2 | 0:25 | Click **robot-a**. **Patrol 50 steps**, **Sync now**. Point at the decision feed and the header. | "Robots revisit the same places all day. Repeats are deduplicated on the device: same thing, same place. Only surprises cross the wire. Header: bytes actually sent vs sending everything." |
| 3 | 0:50 | Cut robot-a's network. **Report oil spill**, **Private note**. Ask `oil spill`. | "Offline, it still remembers, decides and answers on the device in milliseconds. The private note never leaves it." |
| 4 | 1:10 | Restore, **Sync now**, point at the sync diff. Then robot-b: ask `oil spill`, cut its network, ask again. | "The outbox drains and the diff proves convergence. Robot B never saw the spill, so it escalates to the cloud, caches the answer, and now knows it offline." |
| 5 | 1:40 | Point at the **Qdrant Edge inspector**. | "Every Qdrant call in the fleet, live: dense HNSW, built-in BM25, payload filters on Hilbert cells, decay formulas, MMR, facets, with measured latency." |
| 6 | 1:55 | robot-c: **Drop uplink 20 s**. Wait for "no heartbeat", then for "online". | "A real outage on a real process: it keeps working, its heartbeat stops, and it delivers its backlog by itself when the link returns. You could kill -9 it; the test suite does." |
| 7 | 2:25 | Cut both simulators, **Both robots see the toolbox**, restore, sync each twice. Show the inbox. | "Two robots saw the same toolbox offline. The cloud decides by place and time: merge duplicates, keep moves, and ask a human when unsure." |
| 8 | 2:45 | Scroll to **Evidence**. | "Measured, not claimed: results on public human-labelled datasets, including where we are weak (the gate answers 12% of unanswerable questions), and a 12-check stress test. One command each." |

## Questions you will probably get

* **"Is this using a real Qdrant Server?"** By default the cloud is a local stand-in because a
  server is not guaranteed at the venue. `QdrantServerCloud` (`LOCI_QDRANT_URL=...`,
  `make ui-server`) is built and tested on qdrant-client's in-process engine and against HTTP error
  paths; the header chip shows which cloud is live. Say plainly: not yet run against a live server
  by us. With Docker at the venue: `make qdrant-up ui-server`.
* **"Are the devices real?"** robot-c and robot-d are separate OS processes with their own data
  directories, talking HTTP; `python -m loci.edge.node` runs one on another machine
  (`LOCI_CLOUD_TOKEN` secures it). Their sensors are simulated.
* **"Is the data real?"** The retrieval, dedupe and abstention numbers are on public datasets with
  human labels (STS-B, MSRP, SICK). The bandwidth numbers use synthetic patrols, because they need
  ground truth about which object was where. The default embedder is a labelled stand-in.
* **"Why not just dedupe by similarity?"** Look-alikes in different places, or an object that
  moved, get merged and an event is lost: place-blind dedupe loses 12% of events at the same
  setting, and on real text a similarity threshold alone confuses related but different sentences.
* **"Does the LLM see private data?"** No. It is optional, runs only in the cloud over memories the
  cloud already holds, and has a rule-based fallback. Tested against a fake endpoint only.
