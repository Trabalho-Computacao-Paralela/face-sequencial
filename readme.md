# Como executar o código
1. Clone o repositório na sua máquina local (linux)
2. Certifique-se de ter o OpenMP e OpenCV instalados
3. Vá até a raiz do diretório e insira o seguinte comando para compilação:
```
$ g++ geeks.cpp -fopenmp -o output -I /usr/local/include/opencv4 -L /usr/local/lib -lopencv_core -lopencv_objdetect -lopencv_highgui -lopencv_imgproc -lopencv_videoio -fopenmp
```
4. Execute:
```
$ ./output
```
