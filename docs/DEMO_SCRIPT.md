# Three-minute demo script

Setup (once, before the room fills): `make setup`, then `python -m loci.edge.ui --no-autosync` and
open http://127.0.0.1:8765. `--no-autosync` means the only syncs are the ones you click, so the
story below is exactly what happens. Fallback if anything misbehaves: play
[assets/edge-demo.webm](assets/edge-demo.webm) (about 90 s) and talk over it.

Say up front: **"Everything you will see is synthetic warehouse data. The Qdrant Edge shards, the
policy, the sync and the measurements are real."**

## Beats

| # | Time | Do | Say |
|---|---|---|---|
| 1 | 0:00 | Robot A tab. Click **Patrol 50 steps**, then **Sync now**. Point at the decision feed. | "A robot revisits the same five spots all day. Repeats are deduplicated on the device: same thing, same place. Only surprises cross the wire. Header: bytes sent vs. naive sync." |
| 2 | 0:40 | Click the big network button (goes OFFLINE). **Patrol 10**, **Report oil spill**, **Private note**. Ask `oil spill`. | "Network gone. It still remembers, decides and answers, entirely on the device (see the OFFLINE label and the millisecond latency). The private note is pinned to the device." |
| 3 | 1:10 | Restore the network, **Sync now**. Point at the sync diff ("converged") and the header. | "Reconnect: the outbox drains, and the diff proves both sides agree. Retries survive restarts; sends are idempotent." |
| 4 | 1:30 | Robot B tab. Ask `oil spill`. Point at ESCALATED TO CLOUD. Cut B's network, ask again. | "Robot B never saw the spill, so it isn't confident locally and asks the cloud. The answer is cached in its mirror, so it now works offline. When neither is confident it abstains rather than guess." |
| 5 | 2:00 | Cut both networks. **Both robots see the toolbox**. Restore both, **Sync now** on each (twice). Show the inbox. | "Two robots saw the same toolbox while offline. The cloud decides by place and time: same place and window means merge; moved means keep both and the newest is where it is now." |
| 6 | 2:30 | **Blurry toolbox view**, sync both. Click **Same object: merge**. Ask `banana submarine`. | "A blurry view is similar but not certain, so nothing is auto-merged: a human decides. And junk questions get no answer." |
| 7 | 2:50 | Show `docs/assets/edge-hero.svg` or run `make verify`. | "Every claim is re-measured by one command: place-aware dedupe keeps 100% of events at about 15% of the bytes; a place-blind similarity threshold loses 12% of them at the same setting." |

## Questions you will probably get

* **"Is this using a real Qdrant Server?"** Not yet. Sync targets a local stand-in behind a small
  `CloudStore` interface, because a server is not guaranteed at the venue. A server client and
  partial-snapshot pull are the known next step; say so plainly.
* **"Is the data real?"** No. Seeded synthetic vectors and a hashed bag-of-words embedder. The
  verification harness is how to re-measure with a real model.
* **"Why not just dedupe by similarity?"** Two visually similar objects in different places, or one
  object that moved, get merged and an event is lost. The chart in the README shows the cost.
* **"Does the LLM see private data?"** No. It runs only in the cloud, over memories the cloud
  already holds; private memories never leave the device. It is optional and has a deterministic
  fallback. It has been tested against a fake local endpoint, not a live provider.
* **"What happens with look-alikes at the same spot?"** They cannot be told apart from noisy
  re-views by similarity alone, so the default sends them and the cloud asks a human.
