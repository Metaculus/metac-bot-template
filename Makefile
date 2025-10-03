ctx:
	python tools/context_pack.py --base origin/main

ctx-changed:
	python tools/context_pack.py --base $(shell git describe --tags --abbrev=0 2>/dev/null || echo origin/main)
