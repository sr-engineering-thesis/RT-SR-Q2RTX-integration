all:
	gcc -o test.so test.c -shared -fPIC -std=c99 -Wall
