# PythonLambdaDockerECR
PythonLambdaDockerECR

* Learn how to run lambda on docker container and deploy Base image on ECR and use on Lambda 


#### Commands 
```
docker build -t name .

docker run -p 9000:8080 name:latest

curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{ "url": "https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4"}'

```
