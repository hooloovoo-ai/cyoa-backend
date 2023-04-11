#!/bin/bash

celery -A backend.tts worker --loglevel=INFO --concurrency 1 --pool threads --hostname "tts$CUDA_VISIBLE_DEVICES@%h" -Q tts