docker build -t pytorch180_effdet -f Dockerfile .
docker run -p 8889:8889 -it -v $PWD/:/work -v /media/syokoi/vol1:/volume --ipc=host --rm --gpus all pytorch180_effdet /bin/bash
cd ../work/app
streamlit run app.py --server.port 8889