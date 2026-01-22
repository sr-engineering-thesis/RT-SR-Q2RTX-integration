#include <stdlib.h>
#include <stdio.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

typedef unsigned char byte;

typedef struct {
	int width;
	int height;
	int pitch;
	sem_t mutex;
	sem_t empty;
	sem_t full;
} SharedFrameSynchronization;

SharedFrameSynchronization* pixel_meta_data;
byte* pixel_data;
int counter = 0;
void init () {
    int meta_data_fd = shm_open("/shared_frame_meta_data", O_RDWR, 0666);
    size_t meta_data_size = sizeof(SharedFrameSynchronization);
    pixel_meta_data = mmap(NULL, meta_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, meta_data_fd, 0);
    int frame_fd = shm_open("/shared_frame", O_RDWR, 0666);
    size_t shared_frame_size = pixel_meta_data->pitch * pixel_meta_data->height;
    pixel_data = mmap(NULL, shared_frame_size, PROT_READ | PROT_WRITE, MAP_SHARED, frame_fd, 0);
}

byte *get_frame(){
    return pixel_data;
}

int get_frame_height() {
    return pixel_meta_data->height;
}

int get_frame_width(){
    return pixel_meta_data->width;
}

int get_frame_pitch(){
    return pixel_meta_data->pitch;
}

void frame_wait() {
    sem_wait(&pixel_meta_data->full);
    sem_wait(&pixel_meta_data->mutex);
}

void frame_post() {
    sem_post(&pixel_meta_data->mutex);
    sem_post(&pixel_meta_data->empty);
}
