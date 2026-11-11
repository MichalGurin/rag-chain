#!/bin/sh
cd "$(dirname "$0")" || exit 1
./start-app.sh
status=$?
if [ "$status" -ne 0 ]; then
    printf '\nStartup failed. Press Enter to close this window.'
    read -r _
fi
exit "$status"
