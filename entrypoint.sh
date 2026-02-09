#!/bin/bash
# entrypoint.sh

# Start all supervisor services in the background
/usr/bin/supervisord -c /etc/supervisor/supervisord.conf &

# Wait a moment for services to start
sleep 3

# Now start bash or whatever command was passed
exec "$@"