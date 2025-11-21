#!/bin/bash
cd /home/site/wwwroot
uvicorn app.main:app --host 0.0.0.0 --port 8000
