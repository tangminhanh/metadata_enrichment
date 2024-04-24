FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt ./

# install packages 
RUN yum install -y gcc gcc-c++ make cmake opencv opencv-python-headless  pkgconfig poppler-cpp-devel
RUN pip3 install -r requirements.txt
RUN mkdir /data

COPY model.py ./
CMD ["model.lambda_handler"]
