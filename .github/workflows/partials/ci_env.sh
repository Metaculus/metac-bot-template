#!/usr/bin/env bash
# Common environment flags to keep CI runs side-effect free and quiet.
export DISABLE_GIT_PUSH=1
export RESOLVER_CI=1
export PYTHONWARNINGS=ignore
