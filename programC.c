#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#define QUANT 8

#define CONV_SIZE 3


double w0[1][32][3][3]; //conv
double w1[32][64][3][3]; //conv
double w2[64][128][3][3]; //conv
double w3[2048][512]; //dense
double w4[512][7]; //dense

double y[7] = {0, 0, 0, 0, 0, 0, 0};

int read(FILE *file, char* ch){
	char i = 0;
	while(1){
		ch[i] = fgetc(file);
		if(ch[i] == ' ') break;
		i++;
	}
	ch[i] = 0;
	return atoi(ch);
}
//for unsigned char
void img_read(int size_y, int size_x, char *name, unsigned char n[size_y][size_x]){
	char ch[10];
	int i = 0, j = 0;
	unsigned char count = 0;
	FILE *file;
	
	file = fopen(name, "r");
	if (file){
	    while (i < size_y){
	    	j = 0;
	    	while(j < size_x){
	    		n[i][j] = (unsigned char) read(file, ch);
	    		j++;
			}
			i++;
		}
	}
	fclose(file);
	file = NULL;
}

float readfl(FILE *file, char* ch){
	char i = 0, j = QUANT+2;
	ch[i] = getc(file);
	if(ch[i++] == '-') j++;
	while (i <= j){
		ch[i++] = fgetc(file);
	}
	ch[--i] = 0;
	return atof(ch);
}
//for float
void img_readfl(int size_y, int size_x, char *name, double n[size_y][size_x]){
	char ch[10];
	int i = 0, j = 0;
	unsigned char count = 0;
	FILE *file;
	
	file = fopen(name, "r");
	if (file){
	    while (i < size_y){
	    	j = 0;
	    	while(j < size_x){
	    		n[i][j] = ((double) read(file, ch))/255.0;
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
	double c;
	int k = 0, l = 0;
	unsigned char i = 0, j = 0;
	unsigned char count = 0;
	FILE *file;
	
	file = fopen("Params.txt", "r");
	if (file){
	    while (1){
	    	c = readfl(file, ch);
	    	if(count == 0){
	    		w0[i][j++][k][l] = c;
	    		if(j == 32){
	    			j = 0;
	    			i++;
	    			if(i == 1){
	    				i = 0;
	    				l++;
	    				if(l == 3){
	    					l = 0;
	    					k++;
	    					if(k == 3){
	    						k = 0;
	    						count++;
							}
						}
					}
				}
			}
	    	else if(count == 1){
	    		w1[k][l][i][j++] = c;
	    		if(j == 64){
	    			j = 0;
	    			i++;
	    			if(i == 32){
	    				i = 0;
	    				l++;
	    				if(l == 3){
	    					l = 0;
	    					k++;
	    					if(k == 3){
	    						k = 0;
	    						count++;
							}
						}
					}
				}
			}
	    	else if(count == 2){
	    		w2[k][l][i][j++] = c;
	    		if(j == 128){
	    			j = 0;
	    			i++;
	    			if(i == 64){
	    				i = 0;
	    				l++;
	    				if(l == 3){
	    					l = 0;
	    					k++;
	    					if(k == 3){
	    						k = 0;
	    						count++;
							}
						}
					}
				}
			}
	    	else if(count == 3){
	    		w3[k][l++] = c;
	    		if(l == 512){
	    			l = 0;
	    			k++;
	    			if(k == 2048){
	    				k = 0;
	    				count++;
					}
				}
			}
	    	else if(count == 4){
	    		w4[k][i++] = c;
	    		if(i == 7){
	    			i = 0;
	    			k++;
	    			if(k == 512){
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

void conv2D(int size, double n[size][size], double m[3][3], double p[size - 2][size - 2]){
	int i = 0, j, k, l;
	double temp;
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
			p[i][j] += temp;
			j++;
		}
		i++;
	}
}
void pooling(int size, double n[size][size], int num, double p[size/num][size/num]){
	int i = 0, j;
	double temp;
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

void relum(int size, double n[size][size]){
	int i = 0, j;
	while(i < size){
		j = 0;
		while(j < size){
			if(n[i][j] < 0){
				n[i][j] = 0;
			}
			j++;
		}
		i++;
	}
}

void reluf(int size, double f[size]){
	int i = 0;
	while(i < size){
		if(f[i] < 0){
			f[i] = 0;
		}
		i++;
	}
}

//no need for flattening (it's not Hilbertian, it's direct)
//BUT, it's easier
void flatten(int size, int nums, double n[nums][size][size], double f[nums*size*size]){
	int i = 0, j, k, count = 0;
	while(i < nums){
		j = 0;
		while(j < size){
			k = 0;
			while(k < size){
				f[count] = n[i][j][k];
				k++;
				count++;
			}
			j++;
		}
		i++;
	}
}

void dense(int size, double f[size], int size2, double y[size2], double w[size][size2]){
	int i = 0, j;
	while(i < size2){
		j = 0;
		while(j < size){
			y[i] += f[j]*w[j][i];
			j++;
		}
		i++;
	}
}

void printm(int size, double m[size][size]){
	int i, j;
	for(i = 0; i < size; i++){
		for(j = 0; j < size; j++){
			printf("%f ", m[i][j]);
		}
		printf("\n");
	}
}

void printmm(int size_y, int size_x, double m[size_y][size_x]){
	int i, j;
	for(i = 0; i < size_y; i++){
		for(j = 0; j < size_x; j++){
			printf("%f ", m[i][j]);
		}
		printf("\n");
	}
}
/*
void transpose(int size_y, int size_x, unsigned char m[size_y][size_x]){
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
*/

void test(){
	double a[4][4] = {{1,1,1,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
	double b[3][3] = {{1,1,1},{0,0,0},{-1,-1,-1}};
	double c[1][2][2] = {0};
	double d[1][1] = {0};
	double e[4] = {0};
	double f[4][2] = {{1,-2},{0,0},{3,-1},{0,-2}};
	double g[3] = {0};
	int i;
	
	printf("Convolution of: \n");
	printm(4, a);
	
	printf("With kernel: \n");
	printm(3, b);
	
	conv2D(4, a, b, c[0]);
	printf("Convolution result: \n");
	printm(2, c[0]);
	
	pooling(2, c[0], 2, d);
	printf("Pooling result: \n");
	printm(1,d);
	
	flatten(2, 1, c, e);
	printf("Flattening result: \n");
	i = 0;
	while(i < 4){
		printf("%f ", e[i++]);
	}
	printf("\n");
	
	dense(4, e,  2, g, f);
	printf("Dense with weights: \n");
	printmm(4, 2, f);
	
	printf("Result: \n");
	i = 0;
	while(i < 2){
		printf("%f ", g[i++]);
	}
	printf("\n");
	
	reluf(2,g);
	i = 0;
	printf("After relu: \n");
	while(i < 2){
		printf("%f ", g[i++]);
	}
	printf("\n");
	
}

int main(){
	int i = 0, j, k;
	
	double n[50][50];
	
	double p0[32][48][48] = {0};
	double q0[32][24][24] = {0};
	double p1[64][22][22] = {0};
	double q1[64][11][11] = {0};
	double p2[128][9][9] = {0};
	double q2[128][4][4] = {0};
	double f0[2048] = {0};
	double f1[512] = {1};
	double f2[7] = {2};
	
	img_readfl(50, 50, "Img.txt", n);
	reader();
	
	printf("Starting first convolution layer...\n");
	//CONV1
	i = 0;
	while(i < 32){
		conv2D(50, n, w0[0][i], p0[i]);
		pooling(48, p0[i], 2, q0[i]);
		relum(24, q0[i]);
		i++;
	}
	printf("Starting second convolution layer...\n");
	//CONV2
	i = 0;
	while(i < 32){
		j = 0;
		while(j < 64){
			conv2D(24, q0[i], w1[i][j], p1[j]);
			j++;
		}
		i++;
	}
	
	i = 0;
	while(i < 64){
		pooling(22, p1[i], 2, q1[i]);
		relum(11, q1[i]);
		i++;
	}
	
	printf("Starting third convolution layer...\n");
	//CONV3
	i = 0;
	while(i < 64){
		j = 0;
		while(j < 128){
			conv2D(11, q1[i], w2[i][j], p2[j]);
			j++;
		}
		i++;
	}
	
	i = 0;
	while(i < 128){
		pooling(9, p2[i], 2, q2[i]);
		relum(4, q2[i]);
		i++;
	}
	
	printf("Starting flattening...\n");
	//FLATTEN
	flatten(4, 128, q2, f0);
	
	printf("Starting first dense layer...\n");
	//DENSE1
	dense(2048, f0, 512, f1, w3);
	reluf(512,f1);
	
	printf("Starting final dense layer...\n");
	//DENSE2
	dense(512, f1, 7, f2, w4);
	reluf(7,f2);
	
	printf("Final results: \n");
	//print f2
	i = 0;
	while(i < 7){
		printf("%f ", f2[i++]);
	}
	printf("\n");
	printf("\n");
	
	printf("%f", w3[2047][511]);
	
	return 0;
}
