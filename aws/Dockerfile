FROM public.ecr.aws/lambda/python:3.10

# Install system packages
RUN yum install -y gcc gcc-c++ git cmake

# Copy files
COPY aws/lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY /models/price_predictor.pkl ${LAMBDA_TASK_ROOT}/models/
COPY /data/features.csv ${LAMBDA_TASK_ROOT}/data/

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Set entrypoint
CMD ["lambda_function.lambda_handler"]
