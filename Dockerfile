FROM public.ecr.aws/lambda/python:3.12

WORKDIR ${LAMBDA_TASK_ROOT}

COPY requirements-lambda.txt .

RUN pip install --no-cache-dir \
    -r requirements-lambda.txt

COPY generate_narration.py .
COPY omni_voice.py .
COPY speech_engine.py .
COPY transcript_quality.py .
COPY narration_mastering.py .
COPY spectral_matching.py .
COPY storytelling.py .
COPY story_audio.py .
COPY lambda_function.py .
COPY extractor.py .
COPY narration_script.py .
COPY worker_document.py .
COPY narration_content ./narration_content

CMD ["lambda_function.lambda_handler"]
