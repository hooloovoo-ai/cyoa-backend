#!/bin/bash

celery -A backend.images worker --loglevel=INFO --concurrency 1 --pool threads --hostname "images$PORT@%h" -Q images