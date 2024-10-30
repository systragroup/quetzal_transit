# docker build need to be run from folder containing model and quetzal library
FROM public.ecr.aws/lambda/python:3.12


# Install dependancies and add them to paths
COPY ./requirements.txt .
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
ENV PATH="${PATH}:${LAMBDA_TASK_ROOT}/bin"
ENV PYTHONPATH="${PYTHONPATH}:${LAMBDA_TASK_ROOT}"

# Copy src code
COPY ./${QUETZAL_MODEL_NAME} ${LAMBDA_TASK_ROOT}
COPY ./quetzal-network-editor-backend/main.py ${LAMBDA_TASK_ROOT}

# Entrypoint
CMD ["main.handler"]
