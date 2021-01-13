#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define IMG_SIZE 64
#define A1 4
#define A2 3

#define B1 4

#define C1 (IMG_SIZE - 2) * (IMG_SIZE - 2)
#define C2 7
#define QUANT 4

#define CONV_SIZE 3


float w0[A1][A2][A2];
float w1[B1];
float w2[C1][C2];
float y[C2] = {0, 0, 0, 0, 0, 0, 0};

float read(FILE *file, char* ch){
	char i = 0, j = QUANT+2;
	ch[i] = getc(file);
	if(ch[i++] == '-') j++;
	while (i <= j){
		ch[i++] = fgetc(file);
	}
	ch[--i] = 0;
	return atof(ch);
}

int read_b(FILE *file, char* ch){
	char i = 0;
	while(1){
		ch[i] = fgetc(file);
		if(ch[i] == ' ') break;
		i++;
	}
	ch[i] = 0;
	return atoi(ch);
}

void img_read(int size_x, int size_y, char *name, float n[size_y][size_x]){
	char ch[10];
	float c;
	int k = 0;
	unsigned char i = 0, j = 0;
	unsigned char count = 0;
	FILE *file;
	
	file = fopen(name, "r");
	if (file){
	    while (1){
	    	c = read(file, ch);
	    	n[i][j++] = c;
	    	if(j == size_x){
	    		j = 0;
	    		i++;
	    		if(i == size_y){
	    			break;
				}
			}
		}
	}
	fclose(file);
	file = NULL;
}
//basic - for unsigned char
void img_read_b(int size_y, int size_x, char *name, unsigned char n[size_y][size_x]){
	char ch[10];
	int i = 0, j = 0;
	unsigned char count = 0;
	FILE *file;
	
	file = fopen(name, "r");
	if (file){
	    while (i < size_y){
	    	j = 0;
	    	while(j < size_x){
	    		n[i][j] = (unsigned char) read_b(file, ch);
	    		j++;
			}
			i++;
		}
	}
	fclose(file);
	file = NULL;
}

void reader(){
	char ch[10];
	float c;
	int k = 0;
	unsigned char i = 0, j = 0;
	unsigned char count = 0;
	FILE *file;
	
	file = fopen("Params.txt", "r");
	if (file){
	    while (1){
	    	c = read(file, ch);
	    	if(count == 0){
	    		w0[k][i][j++] = c;
	    		if(j == A2){
	    			j = 0;
	    			i++;
	    			if(i == A2){
	    				i = 0;
	    				k++;
	    				if(k == A1){
	    					k = 0;
	    					count++;
						}
					}
				}
			}
	    	else if(count == 1){
	    		w1[k++] = c;
	    		if(k == B1){
	    			k = 0;
	    			count++;
				}
			}
	    	else if(count == 2){
	    		w2[k][i++] = c;
	    		if(i == C2){
	    			i = 0;
	    			k++;
	    			if(k == C1){
	    				k = 0;
	    				break;
					}
				}
			}
	    	else break;
		}
	}
	fclose(file);
	file = NULL;
}

void conv2D(int size, float n[size][size], float m[CONV_SIZE][CONV_SIZE], float p[size - CONV_SIZE + 1][size - CONV_SIZE + 1]){
	int i = 0, j, k, l;
	float temp;
	while(i < size - CONV_SIZE + 1){
		j = 0;
		while(j < size - CONV_SIZE + 1){
			k = 0;
			temp = 0;
			while(k < CONV_SIZE){
				l = 0;
				while(l < CONV_SIZE){
					temp += n[i + k][j + l] * m[k][l];
					l++;
				}
				k++;
			}
			p[i][j] = temp;
			j++;
		}
		i++;
	}
}

void pooling(int size, float n[size][size], int num, float p[size/num][size/num]){
	char i = 0, j;
	float temp;
	while(i < size){
		j = 0;
		while(j < size){
			if(n[i][j] > n[i][j+1]){
				temp = n[i][j];
			}
			else{
				temp = n[i][j+1];
			}
			if(n[i+1][j] > n[i+1][j+1]){
				if(temp < n[i+1][j]){
					temp = n[i+1][j];
				}
			}
			else{
				if(temp < n[i+1][j+1]){
					temp = n[i+1][j+1];
				}
			}
			p[i/num][j/num] = temp;
			j += num;
		}
		i += num;
	}
}

//no need for flattening (it's not Hilbertian)

void dense(int size, float n[size][size], char k){
	char i, j;
	i = 0;
	while(i < size){
		j = 0;
		while(j < size){
			y[k] += w2[i*size + j][k] * n[i][j];
			j++;
		}
		i++;
	}
}

void printm(int size, float m[size][size]){
	int i, j;
	for(i = 0; i < size; i++){
		for(j = 0; j < size; j++){
			printf("%f ", m[i][j]);
		}
		printf("\n");
	}
}

void printmm(int size_y, int size_x, unsigned char m[size_y][size_x]){
	int i, j;
	for(i = 0; i < size_y; i++){
		for(j = 0; j < size_x; j++){
			printf("%d ", (int)m[i][j]);
		}
		printf("\n");
	}
}
/*
void lin_interpf(int n, int len, float *m, float *y){
	int i = 0;
	float x[n], L;
	L = ((float)len-1)/n;
	while(i < n){
		x[i] = L*((float)i + 0.5);
		i++;
	}
	i = 0;
	while(i < n){
		y[i] = (m[(int)ceil(x[i])] - m[(int)floor(x[i])]) * x[i];//tangent
		y[i] += m[(int)floor(x[i])] - (m[(int)ceil(x[i])] - m[(int)floor(x[i])]) * floor(x[i]);//constant
		i++;
	}
}
*/
void lin_interp(int n, int len, unsigned char *m, unsigned char *y){
	int i = 0;
	float x[n], L;
	L = ((float)len-1)/n;
	while(i < n){
		x[i] = L*((float)i + 0.5);
		i++;
	}
	i = 0;
	while(i < n){
		y[i] = (m[(int)ceil(x[i])] - m[(int)floor(x[i])]) * x[i] + m[(int)floor(x[i])] - (m[(int)ceil(x[i])] - m[(int)floor(x[i])]) * floor(x[i]);
		i++;
	}
}

void lin_interp2(int n, int len, unsigned char m[len][IMG_SIZE+2], unsigned char y[IMG_SIZE+2][IMG_SIZE+2]){
	int i = 0, j;
	float x[n], L;
	L = ((float)len-1)/n;
	while(i < n){
		x[i] = L*((float)i + 0.5);
		i++;
	}
	while(j < len){
		i = 0;
		while(i < n){
			y[i][j] = (m[(int)ceil(x[i])][j] - m[(int)floor(x[i])][j]) * x[i] + m[(int)floor(x[i])][j] - (m[(int)ceil(x[i])][j] - m[(int)floor(x[i])][j]) * floor(x[i]);
			i++;
		}
		j++;
	}
}

void traspose(int size_y, int size_x, unsigned char m[size_y][size_x]){
	int i = 0, j;
	unsigned char temp;
	while(i < size_y){
		while(j <= i){
			temp = m[i][j];
			m[i][j] = m[j][i];
			m[j][i] = temp;
			j++;
		}
		i++;
	}
}

void resize(int len_x, int len_y, unsigned char img[len_y][len_x], unsigned char s_img[IMG_SIZE+2][IMG_SIZE+2]){
	unsigned char temp[len_y][IMG_SIZE+2];
	int i = 0;
	while(i < len_y){
		lin_interp(IMG_SIZE+2, len_x, img[i], temp[i]);
		i++;
	}
	traspose(len_y, IMG_SIZE+2, temp);
	i = 0;
	while(i < IMG_SIZE+2){
		lin_interp(IMG_SIZE+2, len_y, temp[i], s_img[i]);
		i++;
	}
	traspose(IMG_SIZE+2, IMG_SIZE+2, s_img);
}

void grad(unsigned char img[IMG_SIZE+2][IMG_SIZE+2], float s_img[IMG_SIZE][IMG_SIZE]){
	unsigned char s_x[IMG_SIZE - CONV_SIZE + 1][IMG_SIZE - CONV_SIZE + 1], s_y[IMG_SIZE - CONV_SIZE + 1][IMG_SIZE - CONV_SIZE + 1];
	char sobel_x[CONV_SIZE][CONV_SIZE] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
	char sobel_y[CONV_SIZE][CONV_SIZE] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
	
	int i = 0, j;
	unsigned char k, l;
	float temp1, temp2;;
	while(i < IMG_SIZE){
		j = 0;
		while(j < IMG_SIZE){
			k = 0;
			temp1 = 0;
			temp2 = 0;
			while(k < CONV_SIZE){
				l = 0;
				while(l < CONV_SIZE){
					s_img[k][l] = temp1;
					temp1 += (float)img[i + k][j + l] * sobel_x[k][l];
					temp2 += (float)img[i + k][j + l] * sobel_y[k][l];
					l++;
				}
				k++;
			}
			s_img[i][j] = sqrt(temp1*temp1 + temp2*temp2);
			j++;
		}
		i++;
	}
}

void preproces(int len_x, int len_y, unsigned char img[len_y][len_x], float s_img[IMG_SIZE][IMG_SIZE]){
	unsigned char temp[IMG_SIZE+2][IMG_SIZE+2];
	resize(len_x, len_y, img, temp);
	grad(temp, s_img);
}

int main(){
	char i = 0, j;
	float p[A1][IMG_SIZE-2][IMG_SIZE-2], q[A1][IMG_SIZE/2-1][IMG_SIZE/2-1];
	float n[IMG_SIZE][IMG_SIZE] = {};
	unsigned char s_img[4], img[] = {7,5,6,2,4,7,5}, IMG[306][292];
	float S_IMG[IMG_SIZE][IMG_SIZE];
	img_read(IMG_SIZE, IMG_SIZE, "Img_Proc.txt", n);
	reader();
	while(i < A1){
		conv2D(IMG_SIZE, n, w0[i], p[i]);
		pooling(IMG_SIZE/2-1, p[i], 2, q[i]);
		j = 0;
		while(j < C2){
			dense(IMG_SIZE/2-1, q[i], j);
			j++;
		}
		i++;
	}
	
	i = 0;
	while(i < C2){
		if(y[i] < 0) y[i] = 0; //relu
		printf("%f ", y[i++]);
	}
	
	lin_interp(4, 7, img, s_img);
	i = 0;
	while(i < 4){
		printf("%d ", (int)s_img[i++]);
	}
	
	img_read_b(306,292,"Img.txt",IMG);
	//printmm(306, 292, IMG);
	preproces(306, 292, IMG, S_IMG);
	printf("\n");
	//printm(IMG_SIZE,S_IMG);
	
	i = 0;
	while(i < A1){
		conv2D(IMG_SIZE, S_IMG, w0[i], p[i]);
		pooling(IMG_SIZE/2-1, p[i], 2, q[i]);
		j = 0;
		while(j < C2){
			dense(IMG_SIZE/2-1, q[i], j);
			j++;
		}
		i++;
	}
	i = 0;
	while(i < C2){
		if(y[i] < 0) y[i] = 0; //relu
		printf("%f ", y[i++]);
	}
	return 0;
}
