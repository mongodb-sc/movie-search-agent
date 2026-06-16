# Getting the API key

For getting AZURE_OPENAI_API_KEY, go to mdb.link/devrel-passkey and then add the access code shared in https://docs.google.com/document/d/1dvj5uGKMizuOpwMBVYL9ouWIkKvLeCLLWRmwGLuNJMo

After receiving the passkey, run this curl command to get the API key:
curl -X POST "https://vtqjvgchmwcjwsrela2oyhlegu0hwqnw.lambda-url.us-west-2.on.aws/" -H "Content-Type: application/json" -d '{"task":"get_token","data":{"provider":"microsoft","passkey":"<YOUR_PASSKEY>"}}'