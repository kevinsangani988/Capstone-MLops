FROM python:3.10-slim

WORKDIR /app

COPY src /app/src
COPY model /app/model
COPY params.yaml /app/params.yaml
RUN mkdir -p /app/reports
COPY reports/experiment_info.json /app/reports/experiment_info.json

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/src/api/requirements.txt
RUN python -c "import numpy, scipy, sklearn; print('Scientific stack loaded successfully')"

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]