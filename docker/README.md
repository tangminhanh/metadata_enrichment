# PythonLambdaDockerECR
to run lambda on docker container and deploy Base image on ECR and use on Lambda 


#### Commands to test locally
##### prerequisites:
- docker
- install lambda runtime interface emulator. instructions are at:  
   https://github.com/aws/aws-lambda-runtime-interface-emulator?tab=readme-ov-file#getting-started  
##### instructions:
1. build docker image
```
docker build -t name .
```
2. run docker image
```
docker run -p 9000:8080 name:latest
```
3. on another terminal, test with lambda runtime interface emulator.  
   this command passes the url parameter similar to that in lambda functions. u can change if u want to test with other videos👍
```
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{ "url": "https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4"}'

```
