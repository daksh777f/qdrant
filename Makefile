# LOCI Edge: one-command entry points. Nothing here needs Docker or a Qdrant Server.
#
#   make setup    install the package with the edge + UI extras (into the current Python env)
#   make verify   re-measure every headline claim (about 40 s); fails if any check fails
#   make ui       start the mission-control UI at http://127.0.0.1:8765
#   make demo     run the terminal demos (offline slice, then two robots + network split)
#   make test     run the test suite (add BACKEND=edge to run it on Qdrant Edge)
#   make record   re-record docs/assets/edge-demo.webm from the scripted UI walkthrough
#
# Against a real Qdrant Server (needs Docker; not exercised in this repo's CI):
#   make qdrant-up        start qdrant/qdrant on :6333 (data in ./.qdrant-data)
#   make verify-server    run the whole verification harness with Qdrant Server as the cloud
#   make ui-server        mission control with Qdrant Server as the cloud
#   make test-server      run the cloud contract suite against the live server
#   make qdrant-down

PY ?= python

.PHONY: setup verify verify-quick ui demo test record qdrant-up qdrant-down verify-server ui-server test-server

QDRANT_URL ?= http://localhost:6333

setup:
	$(PY) -m pip install -e ".[dev,edge-ui]"

verify:
	$(PY) benchmarks/edge_verify.py

verify-quick:
	$(PY) benchmarks/edge_verify.py --quick --no-write

ui:
	$(PY) -m loci.edge.ui

demo:
	$(PY) examples/edge_p0_slice.py
	@echo
	$(PY) examples/edge_p2_patrol.py

test:
ifeq ($(BACKEND),edge)
	LOCI_TEST_BACKEND=edge $(PY) -m pytest tests -q
else
	$(PY) -m pytest tests -q
endif

record:
	$(PY) scripts/record_demo.py

qdrant-up:
	docker run -d --name loci-qdrant -p 6333:6333 -v $(PWD)/.qdrant-data:/qdrant/storage qdrant/qdrant
	@echo "waiting for Qdrant..."; for i in $$(seq 1 30); do curl -sf $(QDRANT_URL)/readyz >/dev/null && break || sleep 1; done; curl -sf $(QDRANT_URL)/readyz && echo " ready"

qdrant-down:
	docker rm -f loci-qdrant

verify-server:
	LOCI_QDRANT_URL=$(QDRANT_URL) $(PY) benchmarks/edge_verify.py --no-write

ui-server:
	LOCI_QDRANT_URL=$(QDRANT_URL) $(PY) -m loci.edge.ui

test-server:
	LOCI_TEST_QDRANT_URL=$(QDRANT_URL) $(PY) -m pytest tests/test_edge_cloud_contract.py -q
