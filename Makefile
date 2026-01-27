all:
	gcc -o libcomm.so comm.c -shared -fPIC -std=c99 -Wall
