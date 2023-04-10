#!/bin/bash

celery -A backend.generate worker --loglevel=INFO --concurrency 1 --pool threads --hostname "generate$CUDA_VISIBLE_DEVICES@%h" -Q generate