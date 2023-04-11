#!/bin/bash

celery -A backend.general worker --loglevel=INFO