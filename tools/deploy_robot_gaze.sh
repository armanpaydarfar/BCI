#!/usr/bin/env bash
# Single-command wrapper that deploys the freshly built Gaze_Tracking
# binary from the C++ research-interface repo to the lab Harmony robot
# at 192.168.2.1. This wraps the cross-repo deploy script in
# DockerProjects/ubuntu1804_container/HARMONY-UNIT-4/tools/build_and_deploy.sh
# and exists so the operator does not have to remember the underlying
# repo path, the --tool name, or the --push host on the day of the
# experiment.
#
# Run the underlying build_and_deploy.sh directly for advanced cases
# (e.g. pushing all 11 tools, overriding the docker image, or a
# different host). The single-tool scoping below is intentional so a
# stray rebuild does not also clobber the other on-robot binaries.
#
# Outputs you should expect on the terminal come from the underlying
# script:
#   - one "backup: ... -> ....bak.<UTC-ISO-timestamp>" line per
#     existing remote binary (the bak filename is your rollback target)
#   - one "ok: <name> md5=<hash>" line per binary after rsync completes
#   - non-zero exit on md5 mismatch, with both md5s and the bak name
#
# Plan ref: Harmony_Gaze_Calibration_REV00_Plan.md §4.5; see also
# Documents/SoftwareDocs/Reports/Harmony_Gaze_Calibration_CPP_Report.md
# "Deploy script completion (2026-05-19)".

set -euo pipefail

CPP_REPO="/home/arman-admin/DockerProjects/ubuntu1804_container/HARMONY-UNIT-4"
DEPLOY_SCRIPT="${CPP_REPO}/tools/build_and_deploy.sh"
DEPLOY_CONFIG="${CPP_REPO}/tools/deploy.config"
BUILT_BINARY="${CPP_REPO}/dist/01d91ea/Gaze_Tracking"

if [[ ! -f "${DEPLOY_CONFIG}" ]]; then
    echo "deploy_robot_gaze: ${DEPLOY_CONFIG} not found; create from tools/deploy.config.example" >&2
    exit 1
fi

if [[ ! -x "${BUILT_BINARY}" ]]; then
    echo "deploy_robot_gaze: ${BUILT_BINARY} not found; run tools/build_and_deploy.sh first to rebuild" >&2
    exit 1
fi

exec bash "${DEPLOY_SCRIPT}" --push 192.168.2.1 --tool Gaze_Tracking
