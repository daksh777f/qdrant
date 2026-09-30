# LOCI Edge: one-command entry points. Nothing here needs Docker or a Qdrant Server.
#
#   make setup    install the package with the edge + UI extras (into the current Python env)
#   make verify   re-measure every headline claim (about 40 s); fails if any check fails
#   make ui       start the mission-control UI at http://127.0.0.1:8765
#   make demo     run the terminal demos (offline slice, then two robots + network split)
#   make test     run the test suite (add BACKEND=edge to run it on Qdrant Edge)
#   make record   re-record docs/assets/edge-demo.webm from the scripted UI walkthrough

PY ?= python

.PHONY: setup verify verify-quick ui demo test record

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
