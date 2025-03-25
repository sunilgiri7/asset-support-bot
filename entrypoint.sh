#!/bin/bash
set -e

# Wait for the database to be ready
/app/wait-for-it.sh db:5432 -t 30

# Run Django database migrations
python manage.py makemigrations
python manage.py migrate

# Collect static files
python manage.py collectstatic --noinput

python manage.py runserver 0.0.0.0:2025
watchmedo auto-restart --patterns="*.py" --ignore-patterns="*.swp;*~" --recursive --delay=2 -- celery -A asset_support_bot worker --loglevel=info

# Execute the container's main command
exec "$@"
